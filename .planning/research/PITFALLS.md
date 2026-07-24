# PITFALLS — Object Search Exploration

> Companion to `IDEA.md`. This document deliberately does **not** restate the pitfalls already
> covered in IDEA.md §5 (ratio test, SuperPoint frames, soft binning, self-match, low-keypoint
> guard) or §7a (null-vs-zero, abstention, provenance). It covers what those sections *omit*, and
> adds implementation-level specifics — verified numbers, exact API defaults, and code-level
> guards — to what they do cover.
>
> **Verification standard.** Claims marked **[E]** were executed locally against real libraries
> (OpenCV 4.10.0 and 5.0.0, NumPy 1.26/2.0, onnxruntime 1.19.2, SQLite 3.51.0, Python 3.9/3.13,
> macOS arm64). Claims marked **[D]** were read from a primary doc/source (URL in §Sources).
> Anything not so marked is in §Unverified.
>
> Phase numbers refer to IDEA.md §9.

---

## Table of contents

| § | Area | Phases most affected |
|---|------|----------------------|
| 1 | NCC template matching (Method 1) | 2 |
| 2 | Generalized Hough voting, 4-DoF | 5 |
| 3 | RANSAC / geometric model fitting | 5 |
| 4 | DINOv2 dense features | 6, 7 |
| 5 | FastSAM / SAM proposals | 7 |
| 6 | Reproducibility across the system | 1, 2, 3, 5, 6, 7 |
| 7 | SQLite run/rating store | 3 |
| 8 | Statistics: Wilson, Bradley-Terry | 3, 8 |
| 9 | Canvas box drawing | 4 |
| 10 | Cross-cutting traps the brief misses | 1, 3, 4, 8 |

### If you read only six things

Ranked by "will silently produce a wrong scoreboard", not by how interesting they are.

1. **§1.1** — a flat crop makes `TM_CCOEFF_NORMED` return `1.0` **everywhere** on OpenCV 4.x and
   `0.0` everywhere on 5.x. Version-dependent, undocumented, no NaN to catch it. *Phase 2.*
2. **§9.3** — `event.offsetX` is **rounded to an integer**, so the exemplar box can shift by a whole
   image pixel between two identical-looking drags. Defeats every backend determinism guarantee.
   *Phase 4.*
3. **§7.5** — one `COALESCE(wrong_count, 0)` converts every unrated run into a claim of perfect
   precision and recall. The requirement in IDEA.md §7a is one line away from being violated at all
   times. *Phase 3.*
4. **§4.2** — DINOv2-with-registers has **4** extra tokens; `tokens[:, 1:]` shifts the entire feature
   map and yields a plausible-but-wrong similarity map. HuggingFace shipped this exact bug. *Phase 6.*
5. **§1.3 / §1.4** — resizing the template separately from the scene drops the self-match from 1.0000
   to 0.3071, and the spurious noise floor varies 15× with template size, so cross-level `argmax` is
   biased toward the smallest template. Both invalidate a fixed threshold. *Phase 2.*
6. **§2.1** — Lowe's location bin width is `0.25 × max_dim × s`, i.e. **scale-dependent**. The
   parenthetical "(using the predicted scale)" is load-bearing and almost always dropped. *Phase 5.*

And one inversion of conventional wisdom: **§6** — ONNX Runtime thread counts, OpenCV thread counts,
BLAS thread counts, and RANSAC seeding all measured **bit-identical** and need no guard.
`use_deterministic_compute` is a no-op on the CPU EP; `cv2.setNumThreads(1)` is silently ignored.
Spend the effort on §1.8 and §9.3 instead.

---

## 1. NCC template matching (Method 1)

### 1.1 A flat/smooth crop makes **every pixel a perfect match** — and the behaviour differs between OpenCV 4 and OpenCV 5

**Symptom.** The user draws a box on a smooth region (sky, a painted wall, a jersey panel). On
OpenCV 4.x the response map is `1.0` *everywhere*; peak extraction returns a grid of boxes tiling
the whole image, all with score 1.0. On OpenCV 5.x the same crop returns `0.0` everywhere and the
method silently returns nothing. Neither is an error.

**Cause.** `TM_CCOEFF_NORMED` divides by `sqrt(Σ T'² · Σ I'²)` where `T' = T − mean(T)`. A constant
template gives `T' ≡ 0`, so the expression is `0/0`. OpenCV does not raise; it substitutes a value,
and **which value it substitutes changed between major versions**:

| case | OpenCV 4.10.0 | OpenCV 5.0.0 |
|---|---|---|
| flat template, textured image | `min=max=1.0000` | `min=max=0.0000` |
| flat template, flat image | `min=max=1.0000` | `min=max=0.0000` |
| textured template, flat image | `min=max=0.0000` | `min=max=0.0000` |

**[E]** (no NaN or Inf produced in either version). This is not documented **[D]** — the docs give
only the algebraic formula. OpenCV issue #5688 reports the 4.x `1.0` behaviour.

The *near*-flat case is worse than the flat case because no guard triggers:

| template population std (uint8) | max response vs random noise |
|---|---|
| 0.0000 | 0.0000 (v5) / 1.0000 (v4) |
| 0.4359 | 0.2146 |
| 0.6000 | 0.1785 |
| 0.8660 | 0.1940 |

and a near-flat template against a near-flat image reaches `max = 0.9861`, `min = −1.0000` **[E]** —
pure amplified noise that looks like a confident detection.

**Prevention.** Guard *before* calling `matchTemplate`, on the template's own statistics, and make it
a typed diagnostic rather than an exception:

```python
TEMPLATE_STD_FLOOR = 2.0  # uint8 intensity units; tune on the synthetic set

tmpl_std = float(template.astype(np.float64).std())
if tmpl_std < TEMPLATE_STD_FLOOR:
    return SearchResult(
        matches=[],
        diagnostics={"abort_reason": "low_template_variance", "template_std": tmpl_std},
    )
```

Additionally, after every `matchTemplate` call, assert the response map is sane — this catches the
version-behaviour flip if OpenCV changes again:

```python
resp = cv2.matchTemplate(img, tmpl, cv2.TM_CCOEFF_NORMED)
if not np.isfinite(resp).all():
    raise MethodError("NCC produced non-finite response")
if resp.ptp() == 0.0:                        # constant map => degenerate normalisation
    return abort("degenerate_ncc_response", const_value=float(resp.flat[0]))
```

Pin the OpenCV version in `pixi.toml` and add a regression test asserting the observed constant for
that version, so the 4→5 upgrade fails loudly in CI instead of silently changing the scoreboard.

**Phase.** 2 (guard + test), 1 (pin the version, record it in provenance per EVAL-09).

---

### 1.2 The response map is top-left anchored and sized `(H−h+1, W−w+1)`; `minMaxLoc` returns `(x, y)` but NumPy indexing is `(y, x)`

**Symptom.** Every box is offset by half a template, or by the wrong axis entirely — boxes stack
along a transposed diagonal.

**Cause.** Two conventions collide.

- `result.shape == (H − h + 1, W − w + 1)` **[D]** — verified: 200×300 image with a 20×20 template
  gives `(181, 281)` **[E]**. The valid region only; **there is no padding and no centre offset.**
- `cv2.minMaxLoc(result)` returns `maxLoc` as an **`(x, y)`** tuple, whereas
  `np.unravel_index(np.argmax(result), result.shape)` returns **`(row, col) == (y, x)`**. Verified
  on a self-match cropped at `x0=88, y0=37`: `minMaxLoc → (88, 37)`,
  `unravel_index → (37, 88)` **[E]**.

**Prevention.** The response value at index `(r, c)` corresponds to a box whose **top-left corner**
is exactly `(x=c, y=r)` and whose size is the template size. No `+w/2`, no `−1`:

```python
# Canonical, no off-by-one: response index -> xyxy box (exclusive x2/y2)
def resp_index_to_box(r: int, c: int, tw: int, th: int) -> tuple[int, int, int, int]:
    return (c, r, c + tw, r + th)
```

Write one test that crops the template out of the scene at a known `(x0, y0)`, runs NCC, and asserts
`argmax == (y0, x0)` and `score == 1.0` exactly. That single test pins the convention permanently.
Never index a response map with a tuple you did not construct yourself in one of these two forms.

**Phase.** 2.

---

### 1.3 The pyramid trap: resizing the template independently from the image destroys the score

**Symptom.** The pyramid "works" but scores collapse at deeper levels, so an absolute threshold
tuned at level 0 finds nothing anywhere else. Worse, the degradation is **non-monotonic**, so a
threshold sweep looks like noise.

**Cause.** The obvious implementation resizes the crop and the scene separately. The two resamplings
land on different sub-pixel phases, so the downscaled template is no longer a sub-array of the
downscaled scene. Measured on a synthetic textured patch, self-match max response:

| level scale | resize template separately | crop template from the already-downscaled image |
|---|---|---|
| 1.00 | 1.0000 | 1.0000 |
| 0.75 | 0.7976 | 1.0000 |
| 0.50 | **0.3071** | 1.0000 |
| 0.25 | 0.6169 | 1.0000 |

**[E]** — note 0.50 scores *worse* than 0.25.

**Prevention.** Build the pyramid of the **scene** first, then take the template at each level by
cropping the exemplar box (scaled) out of that level's image:

```python
def pyramid_levels(scene: NDArray, box: ExemplarBox, scales: Sequence[float]):
    for s in scales:
        lvl = cv2.resize(scene, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        x0, y0 = int(round(box.x0 * s)), int(round(box.y0 * s))
        tw, th = int(round(box.width * s)), int(round(box.height * s))
        if tw < 8 or th < 8 or lvl.shape[1] < tw or lvl.shape[0] < th:
            continue                        # see 1.4
        yield s, lvl, lvl[y0 : y0 + th, x0 : x0 + tw].copy()
```

This only works because search is confined to a single image (IDEA.md §4) — the exemplar *is* part
of the scene. Note this makes the exemplar self-match exactly `1.0` at every level, so the self-match
labelling requirement (METHOD-04c) applies to Method 1 too, not just Method 2.

**Phase.** 2.

---

### 1.4 Cross-level scores are not comparable: the spurious noise floor scales with template area

**Symptom.** "Take the max response across levels" (IDEA.md §5, Method 1) systematically prefers the
smallest template. Deep pyramid levels win on pure noise and the reported box size is wrong.

**Cause.** `TM_CCOEFF_NORMED` is normalised per-window but not per-*size*: a smaller template has
fewer degrees of freedom, so the maximum over `~(H·W)` independent windows of unrelated content is
much higher. Measured, unrelated template on an 800×600 noise image:

| template | spurious max | 99.9th pct |
|---|---|---|
| 8×8 | **0.5771** | 0.3791 |
| 16×16 | 0.2695 | 0.1910 |
| 32×32 | 0.1427 | 0.0959 |
| 64×64 | 0.0677 | 0.0481 |
| 128×128 | 0.0392 | 0.0243 |

**[E]** — a ~15× spread between 8×8 and 128×128 for pure noise.

**Prevention.** Do **not** compare raw `TM_CCOEFF_NORMED` values across levels or across rotation-bank
entries. Standardise each response map against its own null distribution before the cross-level
`argmax`:

```python
# per-level z-score against that level's own response distribution
med = float(np.median(resp))
mad = float(np.median(np.abs(resp - med))) * 1.4826 + 1e-9
z = (resp - med) / mad
```

Then peak-pick on `z`, and carry the raw score too for the EVAL-08 candidate log. This is precisely
what `search/common/calibration.py` exists for — but the shared calibration module must be invoked
**per pyramid level and per rotation angle**, not once on a merged map. Make `self-similarity`
calibration the default for `ncc`, not an absolute threshold.

**Phase.** 2 (`ncc` + `calibration.py`); revisit in 8 when the threshold sweep exposes the bias.

---

### 1.5 Pyramid level guards: template larger than the level image raises; tiny templates are meaningless

**Symptom.** `cv2.error: ... corr.cols <= img.cols + templ.cols - 1 in function 'crossCorr'` at a
deep level, or a run that returns 40 boxes all sized 3×2.

**Cause.** `matchTemplate` requires `templ.size <= img.size` and raises `cv2.error` otherwise **[E]**.
A zero-area box (a click without a drag) yields a `(0, 0)` template and raises the same way **[E]**.
Nothing warns you that a 4×3 template is statistically worthless (see 1.4).

**Prevention.** Validate at three points. In the Pydantic `ExemplarBox` schema: `width >= 8`,
`height >= 8` (frozen model, per INFRA-08 — this rejects the accidental click at the API boundary).
In the pyramid generator: the `tw < 8 or th < 8 or lvl.shape < (th, tw)` continue-guard in 1.3. And
record the set of levels actually evaluated in `diagnostics`, so a run that silently used only one
level is visible in the UI (METHOD-09).

**Phase.** 1 (schema), 2 (generator + diagnostics).

---

### 1.6 Rotated-template banks: the zero-padding corners dominate the correlation

**Symptom.** With `rotation_angles=[0, 15, 30, ...]`, the rotated entries fire on dark image regions
and score higher than the true instance, or the response scale differs so much between angles that
cross-angle max is meaningless.

**Cause.** `cv2.warpAffine` with `BORDER_CONSTANT` fills the corners of the axis-aligned bounding box
with `borderValue`. Measured for a 70×50 crop:

| angle | output size | fraction of output that is constant padding |
|---|---|---|
| 15° | 80×66 | **0.337** |
| 45° | 84×84 | **0.502** |

**[E]** At 45° **half the template is fabricated constant pixels**, which (a) inflate the template's
apparent variance and (b) correlate strongly with any uniform dark region in the scene.

**Prevention.** Warp a mask alongside the crop and pass it to `matchTemplate`. Masked
`TM_CCOEFF_NORMED` works and produces no NaN in both OpenCV 4.10.0 and 5.0.0, including a mask with
holes, and the self-match argmax is still exact **[E]** — so the historical "masks only for
`TM_SQDIFF`/`TM_CCORR_NORMED`" restriction no longer applies to these versions:

```python
h, w = crop.shape[:2]
M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
cos, sin = abs(M[0, 0]), abs(M[0, 1])
nW, nH = int(h * sin + w * cos), int(h * cos + w * sin)
M[0, 2] += nW / 2 - w / 2
M[1, 2] += nH / 2 - h / 2
rot  = cv2.warpAffine(crop, M, (nW, nH), flags=cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
mask = cv2.warpAffine(np.full((h, w), 255, np.uint8), M, (nW, nH),
                      flags=cv2.INTER_NEAREST, borderValue=0)
mask = cv2.erode(mask, np.ones((3, 3), np.uint8))   # kill the interpolated fringe
resp = cv2.matchTemplate(scene, rot, cv2.TM_CCOEFF_NORMED, mask=mask)
```

The `erode` matters: `INTER_LINEAR` blends real pixels with the zero border along the entire boundary,
so the outermost ring of "valid" pixels is contaminated even where the mask says valid.

Note masked NCC still has a *different* score scale than unmasked (measured max 0.086 vs 0.058 on the
same input **[E]**), so 1.4's per-map calibration applies to the rotation bank as well. Also budget
the cost honestly: 5 pyramid levels × 12 angles = 60 correlations, and the brief already defaults
`rotation_angles=[0]` for that reason.

**Phase.** 2.

---

### 1.7 Channel and dtype traps

**Symptom.** Scores are subtly wrong for uploaded PNGs; or a mysterious assertion failure mentioning
`CV_32F`.

**Cause.** **[E]** all of the following:
- 3-channel image + 3-channel template → a **single-channel** result of the same
  `(H−h+1, W−w+1)` shape; channel contributions are summed. Fine, but the score is not a per-channel
  correlation.
- 4-channel input (PNG with alpha) works **silently**, and the alpha channel contributes to the
  correlation. An uploaded PNG with a uniform opaque alpha adds a zero-variance channel; with a
  non-uniform alpha it adds a phantom feature.
- 3-channel image + 1-channel template **raises** `cv2.error` mentioning `CV_32F` and
  `type == _templ.type()` — confusing, because the real problem is the channel-count mismatch.
- Result dtype is always `float32`.

**Prevention.** Normalise every image at the single ingest point in `api/` — decode with
`cv2.IMREAD_COLOR` (drops alpha, forces 3-channel BGR), assert `img.ndim == 3 and img.shape[2] == 3`
and `img.dtype == np.uint8`, and make that the documented contract of the `SearchMethod` protocol
(it already says `npt.NDArray[np.uint8]`, BGR). Then no method needs a channel guard.

**Phase.** 1 (schema/contract), 3 (upload endpoint, API-06).

---

### 1.8 `matchTemplate` scores depend on the **search-image extent** and the **input dtype** — the single biggest threat to the reproducibility constraint

**Symptom.** Somebody crops the search region as an optimisation ("only search the lower half"), or
tiles a large image, or adds a border. Scores shift in the 7th decimal place. Months later a near-tie
flips and the returned box jumps — and it cannot be reproduced, because "the inputs didn't change".

**Cause.** `matchTemplate` is implemented as block-wise DFT cross-correlation, and the block
decomposition (`blocksize`, `getOptimalDFTSize`, `tileCountX/Y` in
`modules/imgproc/src/templmatch.cpp`) is derived from the **correlation-map size** **[D]**. Change the
search extent and you change the FFT tiling and therefore the rounding. Measured **[E]**: the same
template matched against a full 1080×1920 image versus against `img[300:700, 500:1200]`, compared over
the identical overlapping 301×551 region:

- bitwise identical: **False**
- differing float32 values: **120,678 of 165,851 (73%)**
- max abs diff: **4.17e-07**
- peak score: full image `0.9999999403953552` vs sub-image `1.0` — **the peak value itself differs**

Separately, dtype matters: the same 400×400 image as `uint8` vs the identical data as `float32` gives
non-identical results, `max abs diff = 1.27e-06` **[E]**.

Note how this compounds with §1.4: a 1e-7 perturbation is harmless on its own, but the whole point of
the pyramid is to `argmax` across levels whose spurious noise floors sit within a factor of a few of
each other, so ties and near-ties are common — exactly where a 1-ULP difference flips the winner.

**Prevention.** Treat the **entire preprocessing pipeline as part of the config**, with one canonical
path and no optimisation branches:

```python
def preprocess(img_bgr: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """The ONLY path from a decoded image to a matchTemplate input.
    Search extent, dtype, channel order and memory layout are all part of the
    method's identity -- changing any of them changes the numbers."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)   # uint8; do NOT astype
    return np.ascontiguousarray(gray)                   # pin layout too
```

Never search a sub-region "for speed" without bumping `method_version` (EVAL-09). And never rank on
raw float32 near-ties — quantise first, which makes the ordering robust to the whole class of 1e-7
perturbations:

```python
scores_q = np.round(scores.astype(np.float64), 6)   # noise floor is ~4e-7; 1e-6 clears it
best = int(np.argmax(scores_q))                     # first-occurrence tie-break (§6.3)
```

**Phase.** 2 (`ncc` + the canonical preprocess), 1 (`method_version` in provenance).

---

## 2. Generalized Hough voting in 4-DoF pose space

### 2.1 Lowe's bin sizes, verified verbatim — and the one everyone implements wrong

**The brief's numbers are correct.** Verbatim from Lowe, IJCV 2004, **§7.3** (not §7 — cite the
subsection) **[D]**:

> "Therefore, we use broad bin sizes of 30 degrees for orientation, a factor of 2 for scale, and
> 0.25 times the maximum projected training image dimension (using the predicted scale) for
> location. To avoid the problem of boundary effects in bin assignment, each keypoint match votes
> for the 2 closest bins in each dimension, giving a total of 16 entries for each hypothesis and
> further broadening the pose range."

And §7.4 **[D]**: *"The Hough transform is used to identify all clusters with at least 3 entries in a
bin."* / *"At least 3 matches are needed to provide a solution."*

**Symptom of the trap.** Peaks fragment at large scale ratios; instances at 2× the exemplar scale
never accumulate 3 votes even though the correspondences are correct.

**Cause.** The parenthetical **"(using the predicted scale)"** is load-bearing and almost universally
dropped. The location bin width is **not a constant number of pixels** — it is `0.25 ×
max(exemplar_w, exemplar_h) × s`, where `s` is that vote's own predicted scale. A vote predicting a
2× instance must land in a location bin twice as wide. With a fixed pixel bin width, large-scale
hypotheses get spread over 4× more bins than small-scale ones and never clear the vote floor. Lowe
calls this out explicitly in §7.3 **[D]**: *"it is difficult to compute the range of possible bin
values due to their mutual dependence (for example, the dependency of location discretization on the
selected scale)."*

**Prevention.** Bin location in **units of the predicted object size**, which makes the dependence
disappear:

```python
LOC_BIN_FRAC   = 0.25          # Lowe 2004 §7.3
SCALE_BIN_LOG2 = 1.0           # "a factor of 2"  => log2 bin width 1.0
THETA_BIN_DEG  = 30.0          # => 12 bins

def bin_coords(dx, dy, s, theta_deg, exemplar_max_dim):
    loc_bin_px = LOC_BIN_FRAC * exemplar_max_dim * s     # scale-dependent!
    return (dx / loc_bin_px,
            dy / loc_bin_px,
            math.log2(s) / SCALE_BIN_LOG2,
            (theta_deg % 360.0) / THETA_BIN_DEG)
```

Lowe's own answer to the dense-array problem is the same one to use here **[D]**: *"These problems
can be avoided by using a pseudo-random hash function of the bin values to insert votes into a
one-dimensional hash table, in which collisions are easily detected."* In Python this is simply a
`collections.defaultdict(float)` keyed by the 4-tuple of integer bin indices — never allocate a dense
4-D array. A dense array over ±image size × 12 scale bins × 12 theta bins is both enormous and mostly
empty.

**Phase.** 5.

---

### 2.2 Theta wraps; a naive linear binning splits the most common case (θ ≈ 0) in half

**Symptom.** Un-rotated instances — the majority case — fail to form a peak, while rotated ones work.
Or an `IndexError` on bin index 12 of 12.

**Cause.** With `nb = 12` bins of 30°, soft binning puts each vote in the two nearest bins. For
θ = 355° the two nearest bin centres are bin 11 and bin **12**, which either overflows the array or
becomes a distinct bin from bin 0 — so votes at 355° and 5° never combine even though they are 10°
apart. Verified soft-binning behaviour with correct wrapping:

| θ input | wrapped | continuous bin | soft bins (index, weight) |
|---|---|---|---|
| 0.0 | 0.0 | 0.00 | (11, 0.500), (0, 0.500) |
| 5.0 | 5.0 | 0.17 | (11, 0.333), (0, 0.667) |
| 355.0 | 355.0 | 11.83 | (11, 0.667), (0, 0.333) |
| −5.0 | 355.0 | 11.83 | (11, 0.667), (0, 0.333) |
| 359.9 | 359.9 | 12.00 | (11, 0.503), (0, 0.497) |
| 180.0 | 180.0 | 6.00 | (5, 0.500), (6, 0.500) |

**[E]** Note θ = 0 legitimately splits 50/50 across bins 11 and 0 — that is correct *given wrapping*
and catastrophic without it, because the single most common pose in a repeated-instance image sits
exactly on a bin boundary.

**Prevention.** One helper, used for all four dimensions, with `wrap` only for θ:

```python
def soft_bins(c: float, n_bins: int | None) -> list[tuple[int, float]]:
    """c = continuous bin coordinate. n_bins set => circular wrap, else linear."""
    b0 = math.floor(c - 0.5)
    frac = (c - 0.5) - b0
    pairs = [(b0, 1.0 - frac), (b0 + 1, frac)]
    if n_bins is not None:
        pairs = [(b % n_bins, w) for b, w in pairs]
    return [(int(b), w) for b, w in pairs if w > 0.0]
```

`dx`, `dy`, `log2 s` pass `n_bins=None`; θ passes `n_bins=12`. Test the wrap explicitly: assert that
θ = 359° and θ = 1° share a bin with non-zero weight.

Second-order: `log s` must also be signed-symmetric. Use `log2(s)` (not `log(s)`) so the bin width is
literally `1.0` per Lowe's "factor of 2", and centre bin 0 on `s = 1` so the identity
transform — the exemplar self-match — lands mid-bin rather than on a boundary.

**Phase.** 5.

---

### 2.3 Soft binning is 16 votes per correspondence, and the vote floor must be in *weight*, not *count*

**Symptom.** With soft binning enabled, the "≥3 votes" floor from IDEA.md §5 (2c) is met by a single
correspondence, so every bin becomes a hypothesis and per-peak RANSAC runs hundreds of times.

**Cause.** `2⁴ = 16` bins per vote **[D]**. If you increment an integer counter, one correspondence
contributes `1` to sixteen different bins, and three *spurious* correspondences in nearby poses reach
`3` in a shared bin trivially. Lowe's "at least 3 entries" is 3 *distinct matches*, not 3 increments.

**Prevention.** Accumulate **fractional weight** (product of the four per-dimension weights, which
sums to exactly 1.0 per correspondence) **and** the set of contributing correspondence ids:

```python
bins: dict[tuple[int, ...], float] = defaultdict(float)
members: dict[tuple[int, ...], set[int]] = defaultdict(set)

for corr_id, (dx, dy, s, th) in enumerate(votes):
    cx, cy, cs, ct = bin_coords(dx, dy, s, th, exemplar_max_dim)
    for bx, wx in soft_bins(cx, None):
        for by, wy in soft_bins(cy, None):
            for bs, ws in soft_bins(cs, None):
                for bt, wt in soft_bins(ct, N_THETA_BINS):
                    key = (bx, by, bs, bt)
                    bins[key] += wx * wy * ws * wt
                    members[key].add(corr_id)
```

Then the acceptance test is `len(members[key]) >= 3` (Lowe's rule, matches per bin) *and*
`bins[key] >= vote_weight_floor`. Assert in a test that
`sum(bins.values()) == pytest.approx(len(votes))` — a cheap invariant that catches every weighting
bug at once.

Also: `members[key]` must be a **set of ints**, but never *iterate* it without sorting — see §6.4.

**Phase.** 5.

---

### 2.4 Adjacent bins are one peak; naive enumeration double-counts every instance ~16×

**Symptom.** `sparse-geo` reports 90 detections for 6 instances, in tight clusters. Phase 5's success
criterion ("multiple distinct geometric models, not one") passes for the wrong reason.

**Cause.** Soft binning deliberately smears each hypothesis across 16 adjacent bins, so a single true
instance produces a contiguous 2×2×2×2 block of populated bins, all above the floor. Enumerating
"every bin above the floor" returns that whole block. Per-peak RANSAC then fits 16 nearly identical
models to overlapping correspondence subsets, and NMS afterwards can't fully clean it up because the
boxes are genuinely near-identical and each carries independent inlier evidence.

**Prevention.** Non-maximum suppression **in bin space, before RANSAC** — a greedy sweep in descending
weight that claims a 4-D neighbourhood:

```python
def enumerate_peaks(bins, members, floor_w, min_members=3):
    peaks = []
    claimed: set[tuple[int, ...]] = set()
    # deterministic order: weight desc, then the key itself as tie-break (see 6.3)
    for key, w in sorted(bins.items(), key=lambda kv: (-kv[1], kv[0])):
        if w < floor_w or len(members[key]) < min_members or key in claimed:
            continue
        # claim the 3^4 = 81 neighbourhood so the soft-binning halo cannot re-peak
        for off in itertools.product((-1, 0, 1), repeat=4):
            claimed.add(tuple(k + o for k, o in zip(key, off)))
        # pool correspondences over the whole neighbourhood -- do NOT fit on one bin
        pooled = set()
        for off in itertools.product((-1, 0, 1), repeat=4):
            pooled |= members.get(tuple(k + o for k, o in zip(key, off)), set())
        peaks.append((key, w, sorted(pooled)))
    return peaks
```

Two details that matter. (a) The θ dimension of the neighbourhood must wrap (`% n_theta_bins`) or the
peak at θ-bin 0 leaks a duplicate at θ-bin 11. (b) **Pool the correspondences over the neighbourhood
before fitting**, otherwise you hand RANSAC the ~1/16 of the evidence that happens to have landed in
the single argmax bin — which is exactly how you end up below the "≥4–6 inliers" acceptance floor and
report an empty result for an instance you actually found.

**Phase.** 5.

---

### 2.5 `pairwise-4dof` cost is quadratic in correspondences, and the cap is not a config nicety

**Symptom.** SuperPoint backend hangs for minutes on a 1080p scene.

**Cause.** With `k = 8` neighbours and a scene that yields thousands of keypoints, the correspondence
count `n` is easily 10⁴; pairs are `n(n−1)/2 ≈ 5×10⁷`, each generating 16 bin insertions → ~10⁹
dict operations. For calibration on the scale of the inputs: SIFT on an 800×600 image produced
**13,039** keypoints, ORB **500** (its `nfeatures` default), AKAZE **2,656** **[E]**.

**Prevention.** Make `pairwise_sample_cap` a required field of the config model with a default around
`20_000` *pairs* (not correspondences), and sample pairs with a seeded `np.random.default_rng(seed)`
so the sampling is reproducible (Constraints §8). Prefer sampling pairs that are *spatially close in
the exemplar* — two keypoints 3 px apart give a numerically terrible scale/rotation estimate, so
enforce a minimum exemplar-space separation (e.g. `>= 0.2 × exemplar_max_dim`) before accepting a
pair. Record the realised pair count and the truncation flag in `diagnostics`.

**Phase.** 5.

---

### 2.6 Backend-specific keypoint yield: ORB returns 1 keypoint where SIFT returns 83, and the low-keypoint guard threshold cannot be shared

**Symptom.** METHOD-04c's "below ~20 exemplar keypoints" diagnostic fires for *every* crop on the ORB
backend and almost never on SIFT. The guard looks broken.

**Cause.** Measured on the same 64×64 textured crop: **SIFT 83, ORB 1, AKAZE 1** keypoints **[E]**.
ORB's defaults are `nfeatures=500, scaleFactor=1.2, nlevels=8, edgeThreshold=31, patchSize=31` **[D]**
— `edgeThreshold=31` discards a 31 px border, which on a 64×64 crop leaves a 2×2 usable region, and
8 pyramid levels at 1.2× mean most levels are smaller than the patch.

SIFT yield vs crop size on textured content **[E]**:

| crop | SIFT kp (textured) | SIFT kp (flat) | SIFT kp (linear gradient) |
|---|---|---|---|
| 16×16 | 1 | 0 | 0 |
| 24×24 | 5 | 0 | 0 |
| 32×32 | 10 | 0 | 0 |
| 48×48 | 42 | 0 | 0 |
| 64×64 | 83 | 0 | 0 |
| 96×96 | 201 | 0 | 0 |
| 128×128 | 393 | 0 | 0 |

Two things fall out. The brief's "~20 keypoints" floor corresponds to roughly a **40×40 well-textured
crop** for SIFT. And a **smooth crop yields zero keypoints at every size** — a 128×128 linear-gradient
crop gives 0, same as a 16×16 one. So the guard must be phrased as "too few keypoints", never as "crop
too small"; a large smooth crop is exactly as unusable.

**Prevention.** Make the low-keypoint floor a **per-backend** config field, not a shared constant, and
override ORB's geometry for small crops:

```python
# ORB on a small exemplar: shrink the border and patch, drop pyramid levels
orb = cv2.ORB_create(nfeatures=2000, nlevels=4,
                     edgeThreshold=max(4, min(h, w) // 8),
                     patchSize=max(8, min(h, w) // 4))
```

Also **[E]**: all three detectors are bit-deterministic across repeated runs in-process (5 runs,
1 distinct descriptor hash each) — so the detector is not a reproducibility risk. But
`detectAndCompute` returns descriptors as **`None`**, not an empty array, when zero keypoints are
found **[E]** — the `AttributeError` from `descriptors.shape` is the most likely way this method
crashes rather than emitting its diagnostic.

Finally, descriptor norms differ: SIFT is `float32` (`NORM_L2`), ORB and AKAZE are `uint8` binary
(`NORM_HAMMING`) **[E]**. A `BFMatcher(NORM_L2)` on `uint8` binary descriptors runs without error and
produces meaningless distances — pair `normType` with the backend in the same config object so they
cannot drift apart.

**Phase.** 5.

---

## 3. RANSAC for similarity/affine with few correspondences

### 3.1 Which estimator: `estimateAffinePartial2D` is the 4-DoF one, and its minimal sample is 2

**Symptom.** Boxes come back sheared or mirrored despite the model being called a "similarity".

**Cause.** Three OpenCV functions, three different models, three minimal sample sizes. Verified
minimum point counts **[E]**:

| function | DoF / model | n=1 | n=2 | n=3 | n=4 |
|---|---|---|---|---|---|
| `estimateAffinePartial2D` | **4** (rot + uniform scale + translation) | raises | **OK** | OK | OK |
| `estimateAffine2D` | 6 (full affine: shear + aspect) | raises | returns `None` | OK | OK |
| `findHomography` | 8 (projective) | raises | raises | raises | **OK** |

`estimateAffinePartial2D` is the correct choice for the brief's 4-DoF `(Δx, Δy, log s, θ)` pose space,
and `modelPoints = 2` is confirmed in the OpenCV source **[D]**. Defaults from the source:
`confidence = 0.99`, `maxIters = 1000` in the registrator (the Python wrapper's own `maxIters`
default is documented separately — see §Unverified).

**Prevention.** Use `cv2.estimateAffinePartial2D` and say so in the module docstring, with the
consequence spelled out: **shear and aspect degeneracy are impossible by construction** (see 3.3).
Note it returns `(M, inliers)` where `M` is `None` on failure and `inliers` is an `(n, 1)` uint8
array — `inliers.sum()` is the inlier count, and `M is None` must be handled before any
`M[:, :2]` access.

**Phase.** 5.

---

### 3.2 `cv2.setRNGSeed` does **not** seed OpenCV's RANSAC — but RANSAC is already deterministic

**Symptom.** You add `cv2.setRNGSeed(config.seed)` to satisfy the reproducibility constraint
(IDEA.md §8), write a test that changing the seed changes the result, and the test fails. Or you
never test it and ship a config knob that does nothing.

**Cause.** Both estimators construct their own **local, hardcoded** RNG and never consult the global
`theRNG()` that `cv2.setRNGSeed` writes to **[D]** (`modules/calib3d/src/ptsetreg.cpp`):

```cpp
171:  RNG rng((uint64)-1);   // RANSACPointSetRegistrator::run
284:  RNG rng((uint64)-1);   // LMeDSPointSetRegistrator::run
```

This is **deliberate policy**, not an oversight. OpenCV maintainer `vpisarev`, rejecting PR #24933
which tried to make RANSAC honour `theRNG()` **[D]**:

> "The principle of OpenCV is that functions must have **reproducible results**, even the functions
> that use RNG inside. KMeans should return the same labels for the same point cloud, no matter how
> many times you run it. **RANSAC should always return the same subset.**"

Issue #24835 was closed "won't fix" on that basis. The USAC family is the same: `random_generator_state`
defaults to `0` and `isParallel` to `false` in `usac/ransac_solvers.cpp` **[D]**.

Empirically confirmed **[E]**: on a hard problem (40 correspondences, 60% outliers),
`estimateAffinePartial2D` gives **1 distinct result** across 80 sequential calls; identical with
`setRNGSeed(7)` before every call; identical from 4 worker threads; identical across separate
processes; identical with `OMP_NUM_THREADS=1`. Independently reproduced on 200 correspondences with
30% outliers across seeds `{0,1,2,12345}` for `estimateAffinePartial2D`, `findHomography`,
`USAC_DEFAULT` and `USAC_MAGSAC` — one distinct result each **[E]**.

So the good news is that the reproducibility requirement is **already satisfied** for OpenCV's
RANSAC, with no work. The bad news is that a `seed` field on the Method 2 config is a **lie** unless
you implement your own sampling loop.

**Two extra hazards from the same investigation.**

1. **`cv2.theRNG()` is thread-local** — `cv::theRNG() { return getCoreTlsData().rng; }` in
   `modules/core/src/rand.cpp` **[D]**. So `cv2.setRNGSeed(0)` on the main thread does **not** seed
   worker threads. FastAPI runs non-async endpoints in a threadpool (§7.10), so anything that *does*
   read `theRNG()` — `cv2.kmeans` (a plausible future `calibration.py` strategy), `cv2.randu`, and any
   hand-rolled RANSAC — is unseeded in a request handler. Call `cv2.setRNGSeed` **inside** the worker,
   not at import time.
2. **Avoid `USAC_PARALLEL`.** It reads `const int MAX_THREADS = getNumThreads()` and sizes its
   per-thread candidate arrays from it, seeding thread `i` with `state + i` **[D]** — so its result is
   a function of the machine's core count. Use `cv2.RANSAC`, `USAC_DEFAULT`, `USAC_MAGSAC`, or
   `USAC_ACCURATE`, and make that an explicit comment where the method constant is chosen.

Second finding **[E]**: the result *does* differ between OpenCV **4.10.0** and **5.0.0** for identical
input (different model hashes). The OpenCV version therefore belongs in the EVAL-09 provenance record
alongside the git SHA and model hashes, or ratings collected across an environment bump get pooled
and the scoreboard lies in exactly the way §7a warns about.

**Prevention.** Choose one and document it:

- **Preferred:** drop the `seed` field from the `sparse-geo` config for the OpenCV RANSAC path, and
  state in the module docstring that determinism comes from OpenCV's hardcoded RNG. Add a test that
  asserts byte-identical `M` over 20 calls — it will pass, and it will *fail loudly* if a future
  OpenCV makes RANSAC seed-dependent.
- If a seed must be honourable (e.g. for the sequential-RANSAC alternative in METHOD-04b, or the
  `pairwise-4dof` pair sampler), own the sampling: `rng = np.random.default_rng(config.seed)` and a
  hand-rolled minimal-sample loop calling `cv2.estimateAffinePartial2D(..., method=0)` (least squares,
  no RANSAC) on each sample. Then the seed is real.

Record `cv2.__version__` in provenance either way.

**Phase.** 5 (the estimator + test), 1/3 (provenance field).

---

### 3.3 Degeneracy tests: for a 4-DoF similarity, the brief's shear and aspect checks are vacuous

**Symptom.** The degeneracy rejection from IDEA.md §5 (2c) never rejects anything, and implausible
boxes still reach the UI.

**Cause.** `estimateAffinePartial2D` parameterises `[[a, −b, tx], [b, a, ty]]`. Computed metrics for a
true similarity (1.2× at 23°) **[E]**: anisotropy `s_max/s_min = 1.0` exactly, normalised shear
`0.0` exactly. Those two tests **cannot fire**. They only mean something for
`estimateAffine2D` / `findHomography`.

Reference values for the metrics, all from the SVD of the 2×2 linear part `L = M[:, :2]` **[E]**:

| model | det | s_max | s_min | anisotropy | shear |
|---|---|---|---|---|---|
| similarity 1.2× @ 23° | 1.44 | 1.20 | 1.20 | **1.00** | **0.00** |
| shear 0.9 | 1.00 | 1.547 | 0.647 | 2.392 | 0.669 |
| mirror flip | **−1.00** | 1.00 | 1.00 | 1.00 | 0.00 |
| squash (50:1) | 0.02 | 1.00 | 0.02 | 50.0 | 0.00 |

**Prevention.** For the 4-DoF path the meaningful tests are **scale plausibility** and **mirror
rejection**, plus a box-geometry sanity check the brief does not mention:

```python
def similarity_is_plausible(M, cfg) -> tuple[bool, str]:
    L = M[:, :2]
    det = float(np.linalg.det(L))
    if det <= 0.0:                                  # mirror; impossible for a rigid instance
        return False, "mirrored_or_singular"
    s = math.sqrt(det)                              # uniform scale for a 4-DoF similarity
    if not (cfg.min_scale <= s <= cfg.max_scale):   # e.g. 0.25 .. 4.0
        return False, f"scale_out_of_range:{s:.3f}"
    return True, "ok"

def box_is_plausible(box, scene_shape, exemplar_area) -> bool:
    H, W = scene_shape[:2]
    if box.area <= 0 or box.area > 0.5 * H * W:      # one instance shouldn't be half the image
        return False
    if not (0.1 <= box.area / exemplar_area <= 10.0):
        return False
    # reject boxes almost entirely outside the frame
    return intersection_area(box, (0, 0, W, H)) >= 0.5 * box.area
```

Keep the anisotropy and shear tests in `common/` for the day a 6-DoF backend appears, but write in the
docstring that they are inert for `estimateAffinePartial2D`, so nobody reads a passing degeneracy
check as evidence the model is sound. Log the rejection reason per peak into `diagnostics` — "12 peaks
found, 9 rejected as scale_out_of_range" is the single most useful line a practitioner can see.

**Phase.** 5.

---

### 3.4 Refinement silently changes the model class

**Symptom.** A model that passed the similarity degeneracy check produces a visibly sheared
parallelogram overlay.

**Cause.** `estimateAffinePartial2D`'s `refineIters` argument runs a Levenberg–Marquardt refinement
after RANSAC. That refinement stays within the 4-DoF parameterisation, so it is safe — but a common
"improvement" is to re-fit the accepted inliers with `estimateAffine2D` (6-DoF) "for accuracy", which
reintroduces shear and invalidates the earlier check.

**Prevention.** If you re-fit, re-run the plausibility test on the *refined* model, not the RANSAC
one. Better: don't re-fit with a different model class; the whole point of the 4-DoF choice is that
the output box is a rotated rectangle. Assert the model class in a test:
`np.allclose(M[0,0], M[1,1]) and np.allclose(M[0,1], -M[1,0])`.

**Phase.** 5.

---

## 4. DINOv2 dense feature extraction

### 4.1 Non-multiple-of-14 input is **not** an error — it silently crops

**Symptom.** The similarity map is misaligned with the image by a few pixels, worsening toward the
right/bottom edge; or the token count doesn't match `(H//14) * (W//14)` and the reshape raises far
from the cause.

**Cause.** DINOv2's patch size is **14**, not 16 **[D]** (a documented, repeated confusion — HF issue
#34292). `Dinov2PatchEmbeddings.forward` in HF transformers validates **only the channel dimension**;
the `Conv2d(stride=14)` floor-divides the spatial dims with no warning **[D]**. Measured **[E]**:

| input W | `W // 14` | pixels covered | pixels silently dropped |
|---|---|---|---|
| 700 | 50 | 700 | 0 |
| 701 | 50 | 700 | **1** |
| 704 | 50 | 700 | **4** |
| 714 | 51 | 714 | 0 |

Note 700 and 704 produce the *same* 50-column grid from *different* images — so any mapping computed
from the original `W` is wrong by up to 13 px, while a mapping computed from `14 * grid_w` is exact.

(The facebookresearch reference implementation is stricter — community ONNX exports commonly carry an
`assert` that the input height is a multiple of the patch height, which is why exports fail on
arbitrary sizes rather than misbehaving.)

**Prevention.** Resize to an exact multiple of 14 at the inferencer boundary, and make the
`ONNXInferencer` init-time validation (INFRA-09) assert the model's declared spatial input dims are
`% 14 == 0`:

```python
PATCH = 14

def to_patch_grid(h: int, w: int, long_side: int) -> tuple[int, int]:
    """Resize target that preserves aspect ratio and is an exact multiple of PATCH."""
    scale = long_side / max(h, w)
    gh = max(1, round(h * scale / PATCH))
    gw = max(1, round(w * scale / PATCH))
    return gh * PATCH, gw * PATCH
```

Then **derive the grid from the token count, never from the requested size**:

```python
n_patch_tokens = tokens.shape[1] - 1 - n_register_tokens
assert n_patch_tokens == gh_ * gw_, (n_patch_tokens, gh_, gw_)
grid = tokens[:, 1 + n_register_tokens :].reshape(B, gh_, gw_, D)
```

**Phase.** 6 (inferencer + export script), 1 (`ONNXInferencer` shape validation).

---

### 4.2 Register tokens: 4 extra tokens that shift the entire feature map, silently

**Symptom.** The similarity map looks *plausible* but is offset — peaks land consistently up-and-left
of the true instances, by roughly 4 patch positions on the first row and wrapping thereafter.
Precision is mediocre rather than zero, so the bug survives review.

**Cause.** DINOv2-with-registers prepends **4** register tokens. The token layout is
`[CLS | reg×4 | patches]`, and the reference implementation slices accordingly **[D]**:

```python
"x_norm_clstoken":   x_norm[:, 0]
"x_norm_regtokens":  x_norm[:, 1 : self.num_register_tokens + 1]
"x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :]
```

The near-universal `tokens[:, 1:]` therefore includes 4 register tokens, and reshaping a
`(N + 4)`-length sequence into `(gh, gw, D)` either raises (if the arithmetic is checked) or — with
a `-1` in the reshape — **succeeds and rotates the whole map by 4 positions**. This is not a
hypothetical: HuggingFace transformers shipped exactly this bug, including register tokens in the
patch-token mean (issue #37817) **[D]**.

**Prevention.** Never hardcode `1`. Make the register count an explicit, validated field of the
inferencer, discovered at load time and asserted:

```python
@dataclass(frozen=True)
class DINOv2Layout:
    patch: int = 14
    n_cls: int = 1
    n_register: int = 0          # 4 for the *-with-registers checkpoints

    def patch_slice(self) -> slice:
        return slice(self.n_cls + self.n_register, None)
```

At init, run one dummy forward at a known size and assert
`seq_len - n_cls - n_register == (H // 14) * (W // 14)`. If it doesn't hold, raise — the whole point
of INFRA-09 is that a wrong model fails at load, and this is the highest-value instance of it in the
project. Record `n_register` in the run's provenance.

Extra trap: the *plain* DINOv2 checkpoints have 0 registers, DINOv2-with-registers has 4, and DINOv3
differs again. Since Method 3 and Method 5 share one inferencer (§13 Key Decisions), a backbone swap
changes the layout for both at once.

**Phase.** 6 (inferencer), 7 (inherits it).

---

### 4.3 Token centres are at `(i + 0.5) × 14`, not `i × 14`

**Symptom.** Boxes from `dino-dense` are consistently 7 px up and left of the objects — a half-patch
bias that reviewers attribute to "stride-14 coarseness" and accept.

**Cause. [E]** Token `(row=i, col=j)` covers pixels `[i·14, i·14+14) × [j·14, j·14+14)`; its
**top-left** is `(j·14, i·14)` and its **centre** is `(j·14 + 7, i·14 + 7)`. Using the top-left as the
token's position introduces a uniform −7 px bias in both axes, which the bilinear upsample to full
resolution (the v1 mitigation in IDEA.md §5) preserves rather than removes.

**Prevention.** When upsampling the similarity map, use `cv2.resize(..., interpolation=INTER_LINEAR)`
on the grid and rely on OpenCV's half-pixel convention, which already assumes cell centres — do not
hand-roll a `grid[i] -> pixel i*14` mapping. When converting a component's grid-space bbox to pixels,
convert **edges**, not centres:

```python
x0_px = gx0 * PATCH
x1_px = (gx1 + 1) * PATCH      # gx1 inclusive in grid space -> exclusive in pixels
```

Add a test on a synthetic image with a single high-contrast square at a known location asserting the
recovered box centre is within 7 px of truth in both axes.

**Phase.** 6.

---

### 4.4 Prototype construction: mean-pool-then-normalise ≠ normalise-then-mean-pool

**Symptom.** Two reviewers implement "mean-pool the crop's tokens into a prototype" (IDEA.md §5,
Method 3 step 2) and get materially different thresholds.

**Cause.** The two orders produce genuinely different prototypes — measured cosine similarity between
them: **0.9833** **[E]**. That is a large difference in a regime where the operating threshold sits
around 0.6–0.8. Normalise-then-mean weights every token equally; mean-then-normalise weights
high-norm tokens more, and DINOv2's high-norm outlier tokens are exactly the artefact registers were
invented to absorb — so on a no-register checkpoint, mean-then-normalise lets a couple of outlier
tokens dominate the prototype.

By contrast the *numerical* order of the cosine computation is irrelevant: L2-normalise-then-dot vs
dot-then-divide differ by `5.96e-08` max **[E]**, i.e. float32 epsilon.

**Prevention.** Pick **normalise each token, then mean, then normalise the mean**, write it in the
module docstring as a numbered step (per the readability constraint in §3), and unit-test it. Guard
the zero-norm case: normalising a zero vector yields `nan` **[E]**, which then propagates through the
entire similarity map and makes every threshold comparison `False` — an empty result with no
diagnostic.

```python
def l2(x, axis=-1, eps=1e-12):
    return x / np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), eps)

proto = l2(l2(crop_tokens).mean(axis=0))        # tokens: (N, D)
simmap = l2(scene_grid.reshape(-1, D)) @ proto  # (gh*gw,)
assert np.isfinite(simmap).all()
```

Also note this is exactly the weakness the brief already flags in its Method 3 backlog
("many-to-many token similarity … a single prototype loses part structure") — the normalisation order
is a free, one-line partial mitigation available in v1.

**Phase.** 6.

---

### 4.5 Positional-embedding interpolation differs between the reference and HuggingFace implementations

**Symptom.** Features from your ONNX export don't match a reference notebook's, at any resolution
other than the training one. Similarity thresholds tuned against published examples don't transfer.

**Cause.** Two different interpolations, both called "bicubic":

- **facebookresearch/dinov2** `interpolate_pos_encoding`: bicubic, `antialias=self.interpolate_antialias`,
  and a scale offset — `sx = float(w0 + self.interpolate_offset) / M` with
  `interpolate_offset` defaulting to **0.1**, commented in the source as a *"historical kludge: add a
  small number to avoid floating point error in the interpolation"* **[D]**.
- **HuggingFace `modeling_dinov2.py`**: bicubic, **no** antialias, **no** 0.1 offset,
  `align_corners=False` **[D]**.

At the native 224×224 there is nothing to interpolate and both agree. At the high resolutions Method 3
depends on ("run the scene at high input resolution"), they diverge.

**Prevention.** The positional-embedding interpolation is **baked into the ONNX graph at export
time**. So: (a) record which implementation the export came from in `fetch-models` output and in the
run provenance; (b) because `pos_embed` interpolation depends on input size, an export with dynamic
spatial axes may or may not trace the interpolation correctly — export at a **fixed** input size and
validate numerically against the PyTorch reference at that exact size before adopting it
(this is what `library-review` should be gating for the candidate exporters listed in §14);
(c) fix the scene input size in config, so the same image always produces the same tokens.

Preprocessing constants to write into the inferencer docstring, from the reference repo **[D]**:
`IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)`, `IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)`, eval
resize with `InterpolationMode.BICUBIC`. Input is **RGB**, float, normalised — and `cv2.imread` gives
**BGR uint8**, so a missing `cvtColor` silently degrades features without crashing. Assert the channel
order in the inferencer, do not trust the caller.

**Phase.** 6 (export + inferencer), 1 (`fetch-models` provenance).

---

### 4.6 `connectedComponentsWithStats` returns background as label 0 — a full-image false positive

**Symptom.** Every `dino-dense` run reports one enormous match covering the whole image, always with
rank 1. Precision is halved for a one-character reason.

**Cause. [E]** With two blobs in a 50×37 mask, `connectedComponentsWithStats` returns `n = 3` and:

```
label 0 -> [x=0, y=0, w=50, h=37, area=1807]   <-- BACKGROUND
label 1 -> [x=6, y=5,  w=6,  h=5,  area=23]
label 2 -> [x=30,y=20, w=5,  h=4,  area=20]
```

`stats` rows are `[x, y, w, h, area]`, and label 0 is the background with the full-image bounding box.

**Prevention.** `for label in range(1, n):` — and add an assertion that no returned box exceeds
(say) 50% of the image area, which catches this and 3.3's implausible-box case with one guard.

Same experiment also demonstrates the brief's own concern concretely: two seeds placed 1 px apart
(`[5:9, 6:11]` and `[8:10, 10:12]`) merged into a **single** component with `connectivity=8` **[E]**.
That is the "plain NMS merges touching instances" failure mode, occurring inside connected components
before NMS is even reached. Use `connectivity=4` as the default for `dino-dense` (it merges less),
and make `watershed` from `common/peaks.py` the recommended strategy for dense arrays — which is what
Phase 2's success criterion already asks you to demonstrate.

**Phase.** 6 (uses it), 2 (`peaks.py` provides the alternatives).

---

## 5. FastSAM / SAM "everything mode" proposals

### 5.1 The mask decode order is matmul → **upsample → crop** → threshold at **0.0**

**Symptom.** Masks leak outside their boxes, or are blocky, or come out entirely empty.

**Cause.** The reference Ultralytics ONNX example's `process_mask` is **[D]**:

```python
masks = (masks_in @ protos.float().view(c, -1)).view(-1, mh, mw)
masks = ops.scale_masks(masks[None], shape)[0]     # upsample to image size
masks = ops.crop_mask(masks, bboxes)               # THEN crop to each box
return masks.gt_(0.0)                              # threshold logits at 0.0
```

Two things trip people up. First the order: **upsample then crop**. Ultralytics also ships a faster
variant that crops in 160×160 proto space *before* upsampling; that one is cheaper and produces
visibly jagged, box-clipped masks. Second, the threshold is **`> 0.0` on raw logits** — there is no
sigmoid. `sigmoid(x) > 0.5` is equivalent, but the widely-repeated "apply sigmoid then threshold at
0.5" description leads people to write `sigmoid(x) > 0.0`, which returns **everything as foreground**
(sigmoid output is always > 0).

**Prevention.** Write the decode as numbered steps in the inferencer docstring (Constraints §8
requires exactly this), and unit-test the shape contract at each step. YOLOv8-seg conventions to
assert at load time via INFRA-09: `output0` is detections `(1, 4 + nc + 32, n_anchors)` (transposed
relative to what most code wants), `output1` is prototypes `(1, 32, imgsz/4, imgsz/4)`; the 32
prototype channels are the `k = 32` default shared by YOLACT and YOLOv8-seg **[D]**. With FastSAM's
`imgsz = 1024` **[D]** the proto grid is **256×256**, not the 160×160 of the stock 640-input model —
a hardcoded 160 is a silent shape bug.

Since Method 5 only needs **boxes** (IDEA.md §4: "boxes are the output contract"), the cheapest
correct implementation **skips mask decoding entirely** and uses the detection boxes as proposals.
Decode masks only for the `diagnostics` overlay and for the backlog item "region embedding with
background masked out". Say so in the module docstring — it removes the whole class of bug above from
the v1 critical path.

**Phase.** 7.

---

### 5.2 FastSAM defaults are not YOLOv8 defaults, and `max_det` caps your recall

**Symptom.** Method 5 finds at most a fixed number of instances regardless of image content — and on
a dense lattice image (shelf, PCB, tiles: exactly DOC-01's demo set) it plateaus.

**Cause.** The documented FastSAM inference call uses `imgsz=1024, conf=0.4, iou=0.9,
retina_masks=True` **[D]** — different from stock YOLOv8 (`imgsz=640, conf=0.25, iou=0.7`). And
YOLOv8's `max_det` default is **300** proposals per image (see §Unverified for the exact provenance
of this number). A tile image with 400 tiles is capped before retrieval even runs.

`conf=0.4` is also a high floor for a *class-agnostic proposal* stage: you want over-generation here
and to let the DINOv2 retrieval step do the discriminating. And `iou=0.9` is deliberately permissive
NMS — it keeps heavily-overlapping proposals, which is right for proposals but means the raw proposal
set contains many near-duplicates of the same object (the over-segmentation the FastSAM paper
acknowledges: *"too many sampling points may cause slightly different parts of the object to be
incorrectly segmented as separate masks"* **[D]**).

**Prevention.** Surface `conf`, `iou`, `max_det`, `imgsz` as explicit fields on the
`propose-retrieve` config model with the FastSAM defaults as documented defaults, and record the
**realised proposal count** plus a `proposals_truncated: bool` flag in `diagnostics`. Then EVAL-10's
slice metadata can answer "did Method 5 lose recall because retrieval failed, or because the proposer
capped out?" — which is otherwise unanswerable from the logs. Lower `conf` to ~0.1 and raise `max_det`
for the lattice demo images and treat that as a legitimate per-image config, recorded per run.

Also note the proposal count drives the latency breakdown (EVAL-11): the brief predicts SAM will
dominate Method 5's runtime, and `imgsz=1024` on CPU is why.

**Phase.** 7.

---

### 5.3 Determinism and the letterbox round-trip

**Symptom.** Proposal boxes are offset by a constant few pixels, or squashed, on non-square images.

**Cause.** YOLO preprocessing letterboxes to a square `imgsz` with grey padding; the inverse
transform needs the same `gain` and `pad` used on the way in **[D]** (`ops.scale_boxes`). Recomputing
`gain` from the output size instead of remembering the input's is the classic error, and it only shows
on non-square inputs — so it passes on the square synthetic images from EVAL-03 and fails on the
basketball broadcast frames.

On determinism: nothing in the FastSAM path is stochastic (unlike SAM's `automatic_mask_generator`,
which samples a point grid — MobileSAM's automatic path, the documented alternative backend, does
sample and *does* need the seed from config). And the ONNX Runtime CPU EP was measured bit-identical
across thread counts (see §6.1), so the proposal set is reproducible provided the letterbox is.

**Prevention.** Return the letterbox parameters from the preprocess function as an explicit typed
value, and make the postprocess take them as a required argument — not recompute them:

```python
@dataclass(frozen=True)
class Letterbox:
    gain: float          # scale applied to the original image
    pad_x: float
    pad_y: float
    src_hw: tuple[int, int]

def undo(box_xyxy, lb: Letterbox):
    x0 = (box_xyxy[0] - lb.pad_x) / lb.gain
    ...
```

Round-trip test: letterbox a known box, un-letterbox it, assert equality within 1 px, on a
**non-square** image with both landscape and portrait aspect.

**Phase.** 7.

---

## 6. Reproducibility across the whole system

The headline finding, and it inverts the usual advice: **threading is almost a non-issue for a
CPU-only ONNX + OpenCV pipeline. What actually breaks "identical results" is (a) browser-side
coordinate rounding and (b) letting the preprocessing or search extent vary.** Measured on macOS
arm64 across two independent environments (OpenCV 4.10.0/5.0.0, NumPy 1.26/2.5.1, onnxruntime
1.19.2/1.27.0):

| candidate source | measured effect | verdict |
|---|---|---|
| **Browser `event.offsetX` rounding** | **`offsetX = 124` where the true value is `123.5`** | **matters most** [E] — §9.3 |
| **`matchTemplate` search extent** | **73% of floats differ; peak value 0.99999994 vs 1.0** | **matters** [E] — §1.8 |
| **`matchTemplate` input dtype** (uint8 vs float32) | max abs diff `1.27e-06` | **matters** [E] — §1.8 |
| **Unpinned library versions** | OpenCV 4.10 vs 5.0: different RANSAC result *and* opposite flat-template NCC constant | **matters** [E] — §6.6 |
| **Omitting `SessionOptions` / `providers`** | default `intra_op=0` is machine-dependent; CoreML EP available by default | **matters** [E] |
| **`enable_mem_pattern=True`** | reported run-to-run divergence on CPU EP | **sometimes matters** [D] — 6.2 |
| **`set` iteration order across processes** | 3 different orders in 3 runs (two envs, independently) | **matters** [E] — §6.4 |
| **NMS tie-breaking on equal scores** | 2 different kept-box sets from 3 input orderings | **matters** [E] — §6.3 |
| **config-hash JSON key order** | different hash for the same config | **matters** [E] — §6.5 |
| `cv2.setNumThreads(1)` | **silently ignored** on the macOS GCD backend (14 → 14) | robustness only [E] — 6.2 |
| `USAC_PARALLEL` | seeds per-thread from `getNumThreads()` → machine-dependent | only if you use it [D] — §3.2 |
| onnxruntime `intra_op_num_threads` ∈ {1,2,3,4,8,14} | bit-identical for MatMul / Conv / ReduceSum / Softmax; `max abs diff = 0.0` | **does not matter** [E] |
| onnxruntime `graph_optimization_level` DISABLE_ALL vs ENABLE_ALL | `max abs diff = 0.0` | does not matter (this graph) [E] |
| `use_deterministic_compute=True` | **no-op on the CPU EP** — zero effect on output | does not matter [E]/[D] — 6.2 |
| `cv2.setNumThreads` on `matchTemplate` output | identical hash at 1 / 4 / 14 threads | does not matter [E] |
| OpenCV RANSAC, repeated / threaded / cross-process / seeded | 1 distinct result every way it was tried | does not matter [E] — §3.2 |
| `OMP_/OPENBLAS_/MKL_/VECLIB_NUM_THREADS` ∈ {1,2,8} | identical hashes for matmul, `np.sum`, dot | does not matter on arm64 [E] |
| SIFT / ORB / AKAZE `detectAndCompute` repeated | 1 distinct descriptor hash over 5 runs each | does not matter [E] |
| `np.argmax` / `cv2.minMaxLoc` on ties | both return the **first** occurrence in C order, and they agree | does not matter [E] |

### 6.1 ONNX Runtime thread count does not change CPU results — stop paying for a guard you don't need

**Symptom (of over-engineering).** `intra_op_num_threads=1` is set "for reproducibility", and CPU
inference is 4–8× slower than it needs to be for every run in the benchmark (EVAL-04).

**Cause.** The threading docs say nothing about determinism **[D]**, so the cautious default is to
serialise. But it is structurally unnecessary: MLAS's SGEMM partitions work **only along M and N,
never along K** (the reduction dimension), and the K-blocking factor is a function of `K` alone, not
of thread count **[D]** (`core/mlas/lib/sgemm.cpp` — `ThreadCountM/ThreadCountN` are set so that one
of them is always 1, and `MlasPartitionWork` splits M and N). Every output element's dot product
therefore accumulates in an identical order regardless of thread count, so GEMM-based ops (MatMul,
Gemm, Conv-via-im2col) are thread-count-invariant **by construction**.

Measured twice, independently:
- Conv(64×3×7×7) → ReduceMean → MatMul(2048×2048), ORT 1.19.2: output md5 identical across
  `intra_op_num_threads` ∈ {1, 2, 4, 8} and across 3 repeats each; `max|diff| = 0.0`; also identical
  across `graph_optimization_level` DISABLE_ALL vs ENABLE_ALL **[E]**.
- MatMul(1024×2048 @ 2048×512) → ReduceSum → ReduceSum → Softmax, and Conv+Relu+Conv on 1×3×224×224,
  ORT 1.27.0: bit-identical SHA-256 at `intra_op` ∈ {1, 2, 3, 4, 8, 14} **[E]**.

**One latent hazard worth knowing.** `providers/cpu/reduction/reduction_ops.cc` selects the
*reduction algorithm* using `ThreadPool::DegreeOfParallelism(...)` in the branch condition **[D]** —
e.g. `fast_shape[0] > DegreeOfParallelism * 16 && max(...) > DegreeOfParallelism * 256` chooses a
"fast" path, else a generic one. So thread count really can select a **different code path** for
`ReduceSum`/`ReduceMean`. In the measured case the two paths agreed bitwise, but that is an accident
of the two implementations, not a guarantee. Pinning the thread count means you never find out.

**Prevention.** Pin thread counts — for latency comparability (EVAL-11) and to sidestep the
reduction-path branch — and pin the **execution provider**, which genuinely does change numbers.
`CoreMLExecutionProvider` is available by default on macOS **[E]**, so omitting `providers` is a live
risk:

```python
so = ort.SessionOptions()
so.intra_op_num_threads = 1     # default is 0 == "ORT decides" == machine-dependent
so.inter_op_num_threads = 1
so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL   # pin, don't inherit
sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])  # NEVER omit
assert sess.get_providers() == ["CPUExecutionProvider"]
```

ORT 1.27.0 defaults for reference **[E]**: `intra_op_num_threads=0`, `inter_op_num_threads=0`,
`execution_mode=ORT_SEQUENTIAL`, `graph_optimization_level=ORT_ENABLE_ALL`,
`use_deterministic_compute=False`, `enable_mem_pattern=True`, `enable_cpu_mem_arena=True`. The `0`s
are the ones to change.

Record `onnxruntime.__version__`, the provider list, and the thread counts in provenance (EVAL-09),
and add a same-input-twice `np.array_equal` assertion against the **real** DINOv2 and FastSAM graphs
in Phases 6 and 7 — the synthetic-graph result above does not generalise for free.

**Phase.** 1 (`ONNXInferencer` base), 3 (`lifespan` session creation, API-07), 6/7 (per-model test).

---

### 6.2 Two flags that mislead: `use_deterministic_compute` is a no-op, and `cv2.setNumThreads(1)` is silently ignored

**Symptom.** You set both, believe the pipeline is pinned, and neither did anything. Later a genuine
nondeterminism shows up and you have no idea which guard failed.

**Cause.** Three separate confirmed facts:

1. **`SessionOptions.use_deterministic_compute` does nothing on the CPU EP.** A code search over
   `microsoft/onnxruntime` finds every non-plumbing consumer in `providers/cuda/*` or
   `orttraining/*` — **no `providers/cpu/*` file references it** **[D]**. `session_options.h` says
   only *"Deterministic compute is likely not as performant. This option is default to false."*
   Empirically, setting it `True` at 1 and 8 threads produced byte-identical output to `False` — zero
   effect **[E]**.
2. **`cv2.setNumThreads(1)` is silently ignored on macOS.** The wheels are built with
   `Parallel framework: GCD`, and only `0` has any effect **[E]**:
   `getNumThreads()` → 14; `setNumThreads(1)` → still 14; `setNumThreads(4)` → still 14;
   `setNumThreads(0)` → 1. Corroborated by OpenCV issues #15277 and #9694.
3. **`enable_mem_pattern=True` (the default) has caused run-to-run divergence on the CPU EP** —
   ORT issue #18672, "Different results of consecutive runs for same input", where the reporter's fix
   was disabling memory-pattern optimisation **[D]**. Related: ORT #28018 reported non-deterministic
   TopK with *default* SessionOptions, closed stale, with the maintainer's advice being "always
   explicitly create SessionOptions."

**Prevention.** Set `use_deterministic_compute = True` if you like — it documents intent — but never
count it as the mechanism. Use `cv2.setNumThreads(0)` and **assert it took**. Turn off
`enable_mem_pattern` (cheap insurance for a local single-user app). Never pass `None` for
`SessionOptions`.

```python
cv2.setNumThreads(0)                       # 1 is IGNORED on the GCD backend
assert cv2.getNumThreads() == 1, f"OpenCV threads not pinned: {cv2.getNumThreads()}"
so.enable_mem_pattern = False              # ORT #18672
so.use_deterministic_compute = True        # intent only; no-op on CPU EP
```

Do **not** touch `cv2.setUseOptimized()`. It made no difference to `matchTemplate` output here **[E]**
but it toggles IPP/optimised code paths and its effect is build-dependent — flipping it is a silent
config change.

**Phase.** 1 (a single `determinism.py` preamble module imported before cv2/ort — see 6.7).

---

### 6.3 NMS with tied scores is order-dependent — a real, reproducible-only-by-accident bug

**Symptom.** Rerunning the identical query returns a different set of boxes. It reproduces only when
two candidates have exactly equal scores — which happens constantly with NCC on synthetic lattice
images (EVAL-03), where instances are pixel-identical and score exactly `1.0`.

**Cause. [E]** Three boxes, two of them tied at score 0.9 and overlapping: iterating the six input
permutations through `cv2.dnn.NMSBoxes` yields **2 distinct kept sets** — whichever tied box came
first survives. Any greedy NMS has this property; the input order comes from wherever your candidates
were enumerated, which can be a `set` (§6.4) or a non-stable sort.

**Prevention.** Impose a **total order** on candidates before any suppression, so ties are broken by
geometry rather than by arrival:

```python
def canonical_order(matches: list[Match]) -> list[Match]:
    # score DESC, then y, x, h, w ASC -- a total order with no ties possible
    return sorted(matches, key=lambda m: (-m.score, m.y0, m.x0, m.y1, m.x1))
```

and use `kind="stable"` for every `np.argsort` on scores. Apply the same canonical order to the
EVAL-08 top-N candidate log, or the offline threshold sweep will silently depend on enumeration
order. Add a test that shuffles the candidate list with a fixed seed and asserts the NMS output is
unchanged — this is the single test that would have caught it.

**Phase.** 2 (`nms.py`, `peaks.py`), and every method that builds a candidate list.

---

### 6.4 `set` iteration order varies **between processes**

**Symptom.** The CLI benchmark (EVAL-04) produces slightly different results from the API for the
same run, or from itself on a rerun — never within one process.

**Cause. [E]** `list({"ncc", "sparse-geo", "dino-dense", "propose-retrieve"})` printed three different
orders in three consecutive interpreter launches. `PYTHONHASHSEED` randomises `str.__hash__` per
process, and `set`/`frozenset` iteration order follows hash order. `dict` preserves insertion order
(3.7+) and is safe; `set` is not.

This is a live risk in this codebase specifically: the Hough `members[key]` sets (§2.3), any
de-duplication of candidates, and the method registry if it is a set.

**Prevention.** Two rules, both enforceable by review:
1. **Never iterate a `set` in a way that affects output order.** `sorted(my_set)` at every
   consumption point. `members[key]` in §2.3 is `set[int]` and is consumed as `sorted(pooled)`.
2. Use `dict` (insertion-ordered) for the registry, and a `list` for ordered collections.

Setting `PYTHONHASHSEED=0` in `pixi.toml` is a *belt*, not a fix — it doesn't help anyone running the
package outside `pixi run`, and it masks the bug instead of removing it. Do both: set it, and sort.

**Phase.** 1 (registry), 5 (Hough), 2 (peaks/NMS).

---

### 6.5 The config hash is not stable unless you make it stable

**Symptom.** EVAL-09 provenance shows two different `config_hash` values for what the practitioner
believes is the same config, so `/stats` splits one method into two rows and every `n` halves.

**Cause. [E]** `hashlib.sha256(json.dumps(cfg).encode())` gives different digests for
`{"b":1,"a":2.0}` and `{"a":2.0,"b":1}`. With `sort_keys=True` they agree. Two further sources of
drift **[E]**:
- Float repr: `json.dumps(0.1 + 0.2)` → `0.30000000000000004` vs `json.dumps(0.3)` → `0.3`. A config
  value arrived at by arithmetic hashes differently from the same value typed literally.
- Int/float: `{"t": 1}` → `{"t": 1}` and `{"t": 1.0}` → `{"t": 1.0}`. Pydantic will coerce `1` to
  `1.0` for a `float` field, so hash the **validated model**, never the raw request JSON.

**Prevention.** One function, used everywhere, hashing the *validated* model:

```python
def config_hash(cfg: BaseModel) -> str:
    payload = json.dumps(
        cfg.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,          # NaN/Infinity are not valid JSON; fail loud
    )
    return hashlib.sha256(payload.encode()).hexdigest()
```

Round floats in the config model with a Pydantic validator (e.g. to 6 decimals) if any config value
can be computed rather than typed. Store the canonical JSON string *alongside* the hash (API-03
already requires "config hash + JSON") so a hash mismatch is debuggable rather than mysterious.

**Phase.** 3.

---

### 6.6 The environment is part of the result — record it

**Symptom.** Ratings from before and after a `pixi update` are pooled, and a method's score moves for
no code reason. Exactly the failure §7a warns about, from a source §7a doesn't list.

**Cause. [E]** Two measured, real divergences that no git SHA captures: OpenCV 4.10.0 vs 5.0.0 give
different `estimateAffinePartial2D` results for identical input, and opposite constants for the
flat-template NCC case (§1.1).

**Prevention.** Extend the EVAL-09 provenance record beyond "git SHA, model file hash, config hash,
method version" to include the environment identity:

```python
class Provenance(BaseModel, frozen=True):
    git_sha: str
    method_version: str
    config_hash: str
    model_hashes: dict[str, str]
    cv2_version: str            # e.g. "4.10.0"  -- changes NCC and RANSAC output
    numpy_version: str
    onnxruntime_version: str
    ort_providers: list[str]    # CoreML vs CPU changes numbers
    python_version: str
    pixi_lock_hash: str         # sha256 of pixi.lock: one field covering the whole env
```

`pixi_lock_hash` is the cheapest high-coverage field: one value that changes whenever any dependency
does. Make `/stats` group by it, or at minimum warn when an aggregate spans more than one.

**Phase.** 1 (schema + the `pixi.lock` hash helper), 3 (persisted with every run).

---

### 6.7 NumPy reduction order depends on memory **layout**, not on threads — and the fix is `float64`

**Symptom.** `crop.sum()` gives a slightly different value for a sliced view than for a copy of the
same data. A similarity-map statistic (mean, std, median-absolute-deviation from §1.4) differs by
~1e-7 depending on whether it was computed on `img[y0:y1, x0:x1]` or on a contiguous copy.

**Cause.** Two facts, both verified. BLAS thread count is **not** the culprit: matmul, `np.sum` over
4e6 float32, and a dot product all hashed identically at `OMP_/OPENBLAS_/MKL_/VECLIB_NUM_THREADS`
∈ {1, 2, 8} **[E]** — reductions don't go through BLAS, and BLAS GEMM partitions M/N not K, same as
MLAS. What *does* vary is the summation algorithm, selected by layout. NumPy docs, verbatim **[D]**:

> "often numpy will use a numerically better approach (**partial pairwise summation**) leading to
> improved precision… This improved precision is **always provided when no `axis` is given**. When
> `axis` is given, it will depend on which axis is summed… the improved precision is only used when
> the summation is along the fast axis in memory."

So `a.sum()` and `a.sum(axis=0).sum()` can differ, and `a.sum(axis=0)` on a C-contiguous array can
differ from the same call on a sliced/transposed view — because "the fast axis in memory" changed. A
crop `img[y0:y1, x0:x1]` **is** a non-contiguous view.

**Prevention.** Two lines, and they subsume the whole class:

```python
crop  = np.ascontiguousarray(img[y0:y1, x0:x1])   # pin layout at every pipeline boundary
total = crop.astype(np.float64).sum()             # accumulate in float64
```

Accumulating in float64 is the highest-leverage single change in this whole section: it moves the
reduction-order noise floor from ~1e-7 to ~1e-16, nine orders of magnitude below anything that could
flip a comparison or a peak ordering. Apply it to every statistic feeding `calibration.py`.

Also verified **[E]**: `np.argmax` returns the **first** occurrence of the maximum in C order
(documented **[D]**: *"the indices corresponding to the first occurrence are returned"*), it is
position-determined rather than insertion-determined, and `cv2.minMaxLoc` **agrees** with it. So the
tie-break itself is safe; the danger is always an upstream 1-ULP difference turning an exact tie into
a strict ordering (§1.8).

**A concrete preamble.** Put this in `src/object_search/determinism.py` and import it before
`cv2`/`numpy`/`onnxruntime` anywhere it matters (the env vars must be set before those extensions
initialise their thread pools):

```python
import os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # no measured effect on arm64; free insurance

# PYTHONHASHSEED cannot be set from inside the process -- it must come from the launcher.
if os.environ.get("PYTHONHASHSEED") != "0":
    logger.warning("PYTHONHASHSEED != 0; set it in the pixi task for full reproducibility")

import cv2
cv2.setNumThreads(0)                        # 1 is silently ignored (6.2)
assert cv2.getNumThreads() == 1
cv2.setRNGSeed(0)                           # thread-local: call again inside worker threads (3.2)
```

Note the env vars are marked **unverified for x86/OpenBLAS** — the measurement was on
arm64/Accelerate. They cost nothing, so set them.

**Phase.** 1 (the module + the pixi task env), 2/5/6/7 (the `ascontiguousarray` + float64 discipline).

---

### 6.8 Diagnostics payloads will blow up the API before anything else does

**Symptom.** The diagnostics toggle (UI-04) hangs the browser tab; `/search` responses take seconds
to serialise; SQLite grows by tens of MB per run.

**Cause. [E]** A raw similarity map is large, and JSON makes it far worse:

| map size | float32 raw | as a JSON array of floats |
|---|---|---|
| 800×600 | 1.9 MB | ~10 MB |
| 1920×1080 | 8.3 MB | ~41 MB |
| 4000×3000 | 48.0 MB | ~240 MB |

The same 1920×1080 map quantised to uint8 and PNG-encoded is **2.08 MB** (≈2.77 MB base64) **[E]**.

**Prevention.** Make the `diagnostics` contract explicit that dense arrays are transported as
**PNG-encoded uint8 images**, not numbers, and that they are served from a separate endpoint rather
than inlined in the `/search` response:

```
GET /runs/{run_id}/diagnostics/similarity.png     -> image/png, cacheable
POST /search  -> SearchResult.diagnostics = {"similarity_map_url": ".../similarity.png",
                                              "sim_min": 0.11, "sim_max": 0.94}
```

Carry `sim_min`/`sim_max` so the UI can label the colour bar and the practitioner can still read
absolute values. Keep the *scalar* diagnostics (Hough peak list, keypoint correspondences, proposal
boxes, rejection reasons) inline — those are small and are what actually explains a failure. And never
persist a dense array into SQLite: store the PNG on disk next to the run and keep a path (this also
avoids the row-overflow problem in §7.6).

**Phase.** 3 (API contract), 4 (overlay), 6/7 (the methods that produce maps).

---

## 7. SQLite run/rating store

> Verified against **SQLite 3.51.0** with **Python 3.9.6**'s `sqlite3`.

### 7.1 `CHECK (wrong_count >= 0)` silently accepts anything when the other side is NULL

**Symptom.** You add `CHECK (wrong_count >= 0)` and later `CHECK (wrong_count <= returned_count)` as
the EVAL-18 validation, and neither ever fires on the rows that matter. `wrong_count = 99` with
`returned_count = NULL` inserts cleanly.

**Cause.** Documented **[D]**: *"If the CHECK expression evaluates to NULL, or any other non-zero
value, it is not a constraint violation."* `NULL >= 0` → NULL → passes. Verified: a row of `(NULL)`
inserts into a column with `CHECK (wrong_count >= 0)`, and `(99, NULL)` inserts into a table with
`CHECK (a <= b)` **[E]**.

**Prevention.** This NULL-passes rule is what *permits* the EVAL-17 nullable counts, so don't fight
it — but write CHECKs that state the NULL branch explicitly, so intent is documented and real bad
values are still caught:

```sql
CREATE TABLE ratings (
  run_id       INTEGER PRIMARY KEY REFERENCES runs(id) ON DELETE CASCADE,
  -- NULL = "not assessed" (EVAL-17). 0 = "assessed, none found".
  wrong_count  INTEGER,
  missed_count INTEGER,
  CHECK (wrong_count  IS NULL OR wrong_count  >= 0),
  CHECK (missed_count IS NULL OR missed_count >= 0),
  -- EVAL-18: 0 <= wrong_count <= R, enforced only when both are known
  CHECK (wrong_count IS NULL OR wrong_count <= (SELECT returned_count FROM runs WHERE id = run_id))
);
```

(The subquery form is not permitted in a CHECK — enforce the `<= R` half in Pydantic or a trigger; see
§Unverified.) Test the constraint: insert `(NULL, NULL)`, `(0, 0)`, `(-1, 0)` and assert only the
third raises `sqlite3.IntegrityError`. Schema constraints on nullable columns are **untrustworthy
until tested**.

**Phase.** 3.

---

### 7.2 `AVG` drops unassessed rows from the *denominator*, so every rate is silently conditional

**Symptom.** `/stats` shows "mean precision 0.90" over 4 runs. Only 2 were assessed. The honest
statement is "0.90 over 2 of 4".

**Cause.** Documented **[D]**: *"The avg() function returns the average value of all non-NULL X within
a group"* and returns NULL if there are no non-NULL inputs. NULL rows are not counted as 0 — they are
not counted at all. Verified: 4 runs, 2 assessed with precisions 0.8 and 1.0 →
`COUNT(*) = 4, COUNT(precision) = 2, AVG(precision) = 0.9` **[E]**.

This is exactly what EVAL-13 demands ("aggregates state which subset each metric was computed over"),
and it is the default behaviour you get *wrong* by omission rather than by commission.

**Prevention.** Never expose a bare `AVG`. Ship the coverage denominator in the same row, and make
the Pydantic response model force the caller to have it:

```sql
CREATE VIEW method_summary AS
SELECT method,
       COUNT(*)                     AS n_runs,
       COUNT(precision)             AS n_precision_assessed,
       COUNT(recall)                AS n_recall_assessed,
       AVG(precision)               AS mean_precision,   -- NULL if n = 0
       AVG(recall)                  AS mean_recall
FROM run_metrics GROUP BY method;
```

```python
class MethodSummary(BaseModel, frozen=True):
    n_runs: int
    n_precision_assessed: int
    mean_precision: float | None      # None, never 0.0
```

Render `None` as an em dash, never as `0`, and always print `n_assessed / n_runs` next to the rate.

**Phase.** 3 (view + schema), 4 (rendering).

---

### 7.3 `SUM` returns NULL over all-NULL; `TOTAL` returns `0.0` — one of these manufactures a perfect score

**Symptom.** "Total wrong detections for this method: 0" for a method where **nothing was ever
assessed**. Indistinguishable from a flawless method.

**Cause.** Documented **[D]**: *"If there are no non-NULL input rows then sum() returns NULL"*, while
*"total() returns 0.0"*. Verified across all three cases **[E]**:

| input | `SUM` | `TOTAL` |
|---|---|---|
| one value 3, three NULLs | 3 | 3.0 |
| all NULL | **NULL** | **0.0** |
| zero rows | **NULL** | **0.0** |

**Prevention.** **Ban `TOTAL()` from this codebase for human-count columns** and add it to the CI
grep-lint in §7.8. Use `SUM`. But note `SUM` is *also* misleading in the mixed case — `SUM = 3`
above ignores three unassessed rows — so always pair it:

```sql
SUM(wrong_count)    AS total_wrong,
COUNT(wrong_count)  AS n_wrong_assessed,
COUNT(*)            AS n_total
```

**Phase.** 3.

---

### 7.4 Integer division truncates precision to 0; division by zero returns NULL and impersonates "not assessed"

**Symptom.** Precision reads `0` for a run with 8 TP out of 10. Recall reads `1` for everything. And
an abstention (R = 0) that *was* assessed shows `precision = NULL`, which the UI labels "not
assessed" — collapsing the very distinction EVAL-12 requires.

**Cause. [E]** In SQLite `/` on two INTEGERs is integer division: `1/2 = 0`, `7/10 = 0`. On a live
view with `R = 10, FP = 2`, `(R-FP)/R` → **`0`** while `CAST(R-FP AS REAL)/R` → **`0.8`**. Separately,
division by zero returns **NULL**, not an error: `1/0 IS NULL`, `0/0 IS NULL` **[E]**. So a run with
`R = 0, FP = 0` (assessed!) yields `TP = 0, precision = NULL` — byte-identical in shape to a genuinely
unassessed row.

**Prevention.** You have **three** states, not two, and the view must emit them explicitly:

```sql
CREATE VIEW run_metrics AS
SELECT
  r.id, r.method,
  r.returned_count                     AS R,
  ra.wrong_count                       AS FP,
  ra.missed_count                      AS FN,
  r.returned_count - ra.wrong_count    AS TP,          -- NULL-propagating (7.5)
  (r.returned_count - ra.wrong_count) + ra.missed_count AS expected,
  CASE WHEN ra.wrong_count IS NULL THEN 'not_assessed'
       WHEN r.returned_count = 0     THEN 'undefined_abstention'   -- EVAL-12
       ELSE 'ok' END                   AS precision_status,
  CAST(r.returned_count - ra.wrong_count AS REAL)
    / NULLIF(r.returned_count, 0)      AS precision,
  CAST(r.returned_count - ra.wrong_count AS REAL)
    / NULLIF((r.returned_count - ra.wrong_count) + ra.missed_count, 0) AS recall
FROM runs r LEFT JOIN ratings ra ON ra.run_id = r.id;
```

`CAST(numerator AS REAL)` (not `* 1.0`, which is easy to lose in a refactor) and explicit
`NULLIF(x, 0)` — behaviourally redundant in SQLite but it documents intent and survives a port to a
database that raises. F1 needs the same: `2*P*R / NULLIF(P+R, 0)`. Add a regression test asserting
`type(row["precision"]) is float` and `0 < row["precision"] < 1` for an 8/10 fixture.

**Phase.** 3.

---

### 7.5 One `COALESCE` destroys the entire EVAL-17 guarantee

**Symptom.** Someone "fixes" the API returning `null` for TP by wrapping it in
`COALESCE(ra.wrong_count, 0)`. Every unrated run instantly reports perfect precision and recall —
precisely the scoreboard lie §7a is built to prevent.

**Cause.** NULL propagates correctly through arithmetic for free: `(10 - NULL) IS NULL` **[E]**,
which is exactly what `TP = R - FP` needs. `COALESCE(NULL, 0)` returns `0` **[E]**, and it is a
one-line, well-intentioned change. Verified end-to-end: a rating row with both counts NULL yields
`FP/FN/TP/precision/recall/expected` all NULL, while `(0, 0)` yields `TP=10, precision=1.0,
recall=1.0` — correctly distinct **[E]**.

**Prevention.** Four layers, because this is the project's defining requirement:

1. **CI grep-lint** on both SQL and Python:
   ```bash
   ! grep -rniE 'coalesce\s*\(\s*(ra\.)?(wrong|missed)_count|ifnull\s*\(\s*(ra\.)?(wrong|missed)_count' sql/ src/
   ! grep -rnE '(wrong|missed)_count\s*(or|\|\|)\s*0' src/
   ! grep -rn 'TOTAL(' sql/
   ```
   Also catch `int(row["wrong_count"] or 0)` and `.get("wrong_count", 0)`.
2. **Pydantic response models must be `int | None` with no default.** With bare `int` a `None` raises
   at serialisation (loud, good). With `int = 0` it silently lies. Never give these fields a default —
   this is the same rule as UI-08, one layer down.
3. **One golden fixture test** with exactly four rows — `(NULL, NULL)`, `(2, 1)`, `(R=0, 0, 0)`,
   `(0, 0)` — asserting the complete serialised payload byte-for-byte including `"wrong_count": null`.
   This single test covers §7.2, §7.4, §7.5 and the Wilson `n = 0` case at once.
4. **`thumbs_up` too:** SQLite has no boolean; `0`, `1`, and `NULL` are three distinct values and
   `NULL` must not become `False`.

**Phase.** 3 (all four), 1 (the grep-lint in CI).

---

### 7.6 NULL-safe comparison, and `NOT IN` over a nullable subquery returns nothing

**Symptom.** "All runs where wrong_count is not 3" returns 1 row of 3. A "runs without ratings" filter
returns zero rows for no visible reason.

**Cause. [E]** `WHERE` keeps only rows where the predicate is *true*; NULL is neither:

| predicate over `v IN (1, NULL, 0)` | rows matched |
|---|---|
| `v = 1` | 1 |
| `v <> 1` | 1 (NULL row dropped) |
| `NOT (v = 1)` | 1 (NULL row dropped) |
| `v IS NOT 1` | **2** (NULL row kept) |
| `v NOT IN (1, 2)` | 1 |
| **`v NOT IN (1, NULL)`** | **0** |

`NULL = NULL` is NULL **[E]**. SQLite supports `IS` / `IS NOT` and `IS [NOT] DISTINCT FROM` **[E]**.

**Prevention.** Use `IS` / `IS NOT` for every comparison against a nullable column. Use `NOT EXISTS`,
never `NOT IN`, for anti-joins:

```sql
-- WRONG: returns 0 rows if the subquery yields any NULL
WHERE run_id NOT IN (SELECT run_id FROM ratings)
-- RIGHT
WHERE NOT EXISTS (SELECT 1 FROM ratings ra WHERE ra.run_id = runs.id)
```

And make the assessed/unassessed choice explicit in the API rather than implicit in a `WHERE`:

```python
class AssessmentFilter(StrEnum):
    ASSESSED_ONLY = "assessed_only"
    UNASSESSED_ONLY = "unassessed_only"
    ALL = "all"
```

**Phase.** 3.

---

### 7.7 `GROUP BY` groups NULLs together; `UNIQUE` treats them as all distinct

**Symptom.** (a) A wrong-count histogram shows one blank/`0` bucket holding every unassessed run.
(b) A `UNIQUE(run_id, rater_id)` constraint permits unlimited duplicate ratings once `rater_id` is
NULL.

**Cause.** Two deliberately inconsistent documented rules **[D]**. `GROUP BY`/`DISTINCT`: NULLs are
**not** distinct and group together — sqlite.org calls the choice *"somewhat arbitrary"*, made for
compatibility with Oracle/PostgreSQL/DB2. `UNIQUE`: *"NULL values are considered distinct from all
other values, including other NULLs."* Verified both: 3 NULL rows → one group of 3; 3 rows of `(NULL)`
all accepted into a UNIQUE column **[E]**.

**Prevention.** Label the NULL bucket explicitly so the frontend cannot render it as `0`:

```sql
SELECT CASE WHEN wrong_count IS NULL THEN 'not assessed'
            ELSE CAST(wrong_count AS TEXT) END AS bucket, COUNT(*)
FROM ratings GROUP BY wrong_count;
```

And: **any column in a UNIQUE constraint must be `NOT NULL`.** For this single-user app make ratings
1:1 with runs via `run_id INTEGER PRIMARY KEY REFERENCES runs(id)` — a PK is implicitly NOT NULL, so
nullability lives only on the *count* columns where EVAL-17 wants it, never on identity columns.

**Phase.** 3.

---

### 7.8 JSON storage: `->` vs `->>`, row overflow, and version floors

**Symptom.** `WHERE cfg -> '$.method' = 'sift'` returns zero rows forever; buckets are named
`"sift"` with literal quotes. Separately, `SELECT id, method FROM runs` gets slow as the table grows
even though it touches no JSON.

**Cause.**
- `->` yields a **JSON** representation (quoted); `->>` yields a **SQL value** (unquoted). Verified on
  `{"m":"a"}`: `json_extract → a`, `cfg -> '$.m' → "a"`, `cfg ->> '$.m' → a` **[E]**. Both operators
  arrived in **SQLite 3.38.0 (2022-02-22)** **[D]**.
- Default `page_size` is **4096** **[E]**, so the max payload kept on a table b-tree leaf page is
  `X = U − 35 = 4061` bytes **[D]**. A top-50 candidate list with scores plus a latency breakdown plus
  provenance exceeds that easily, so every row spills to overflow pages, and a scan wanting only
  `id, method` still walks them.
- `json_extract` on a missing path returns NULL **[E]** — indistinguishable from a stored JSON `null`,
  so never use key presence as a boolean signal for provenance.

**Prevention.**
1. `->>` or `json_extract()` for anything compared, grouped, or returned; reserve `->` for sub-objects.
   Assert the floor at startup:
   ```python
   if sqlite3.sqlite_version_info < (3, 38, 0):
       raise RuntimeError(f"need SQLite >= 3.38.0, got {sqlite3.sqlite_version}")
   ```
2. **Split the fat payloads into a side table on day one** — retrofitting means the 12-step rebuild
   (§7.9):
   ```sql
   CREATE TABLE runs (              -- narrow, always hot
     id INTEGER PRIMARY KEY, created_at TEXT NOT NULL,
     method TEXT NOT NULL, image_id TEXT NOT NULL,
     box_x0 INTEGER NOT NULL, box_y0 INTEGER NOT NULL,
     box_x1 INTEGER NOT NULL, box_y1 INTEGER NOT NULL,
     returned_count INTEGER NOT NULL,
     latency_total_ms REAL NOT NULL,
     config_hash TEXT NOT NULL,
     config_json TEXT NOT NULL CHECK (json_valid(config_json))
   );
   CREATE TABLE run_payloads (      -- fat, fetched only on the detail view
     run_id INTEGER PRIMARY KEY REFERENCES runs(id) ON DELETE CASCADE,
     matches_json TEXT NOT NULL,
     candidates_json TEXT NOT NULL,          -- EVAL-08 top-50
     latency_breakdown_json TEXT NOT NULL,   -- EVAL-11
     provenance_json TEXT NOT NULL,          -- EVAL-09
     slice_meta_json TEXT NOT NULL           -- EVAL-10
   );
   ```
   Declare small hot columns first; column order is a real lever because the record header is
   traversed in order.
3. **`method` is a first-class NOT NULL column, not a JSON path** — it is the grouping key for every
   metric view and every Bradley-Terry comparison.
4. Store **TEXT**, not JSONB. JSONB landed in **3.45.0 (2024-01-15)** and is faster, but the docs say
   to treat it as an **opaque blob** **[D]** — which destroys the human-readable, `.dump`-diffable,
   greppable property that makes `config_json` useful *as* a provenance record. Revisit only if
   profiling shows `json_extract` dominating.
5. Size limits won't save you: `SQLITE_MAX_LENGTH` defaults to **1,000,000,000** bytes **[D]**.
   Enforce domain limits in Pydantic (`Field(max_length=50)` on the candidate list) plus a cheap
   `CHECK (length(candidates_json) < 1000000)`.
6. If you must index a JSON path: **`STORED` generated columns cannot be added by `ALTER TABLE`;
   `VIRTUAL` ones can** **[D]** (generated columns landed in 3.31.0). Or just index the expression:
   `CREATE INDEX ... ON runs(config_json ->> '$.backend')`.

**Phase.** 3.

---

### 7.9 Migrations: `PRAGMA user_version`, and a pure-DDL migration in Python is **not** in a transaction

**Symptom.** A three-statement migration fails on statement 3. `conn.rollback()` is called. The first
two `CREATE TABLE`s are still there and `user_version` no longer matches reality.

**Cause. [E]** Python's legacy transaction control (still the default) opens an implicit `BEGIN` only
before `INSERT`/`UPDATE`/`DELETE`/`REPLACE` — **not** before DDL. Verified: after
`execute("CREATE TABLE a(x)")` on a fresh connection, `conn.in_transaction` is `False`; a subsequent
`CREATE TABLE b` followed by `rollback()` leaves **both** tables. With an explicit
`execute("BEGIN")` first, `rollback()` correctly removes both **[E]**.

`PRAGMA user_version` is a bare signed 32-bit integer at header offset 60, defaulting to **0** **[E]**,
that *"is available to applications to use however they want"* **[D]**. SQLite never validates or
increments it. (Do **not** write to `schema_version`, offset 40 — SQLite owns that one.)

**Prevention.**

```python
MIGRATIONS: list[tuple[int, str]] = [
    (1, "CREATE TABLE runs (...); CREATE TABLE ratings (...);"),
    (2, "ALTER TABLE runs ADD COLUMN cv2_version TEXT;"),
]

def migrate(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA foreign_keys=OFF")          # MUST be outside the transaction
    current = conn.execute("PRAGMA user_version").fetchone()[0]
    if current > max(v for v, _ in MIGRATIONS):
        raise RuntimeError(f"database is newer than this build: v{current}")
    for version, sql in MIGRATIONS:
        if version <= current:
            continue
        conn.execute("BEGIN")                        # MANDATORY for DDL
        try:
            for stmt in filter(None, (s.strip() for s in sql.split(";"))):
                conn.execute(stmt)                   # not executescript(): it auto-commits
            conn.execute(f"PRAGMA user_version = {int(version)}")  # cannot be parameterised
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
    rebuild_views(conn)                              # see below
    conn.execute("PRAGMA foreign_keys=ON")
    assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
```

`ALTER TABLE` supports only `RENAME TO`, `RENAME COLUMN` (3.25.0), `ADD COLUMN`, and `DROP COLUMN`
(3.37.0) **[D]**. There is **no** `ADD CONSTRAINT` — changing a CHECK requires the documented 12-step
table rebuild: FKs off → BEGIN → record indexes/triggers/views from `sqlite_schema` → create
`new_X` → `INSERT ... SELECT` → `DROP TABLE X` → `RENAME new_X TO X` → recreate indexes/triggers →
recreate views → `PRAGMA foreign_key_check` → COMMIT → FKs on **[D]**. `ADD COLUMN` also cannot add
`PRIMARY KEY`/`UNIQUE`, cannot be `NOT NULL` without a non-NULL constant default, and cannot be
`GENERATED ... STORED` **[D]**. So **get the constraints right in migration 1** (§7.1).

Because all derived metrics live in VIEWs (EVAL-07), **view recreation is not optional** — every
rebuild of `runs` or `ratings` invalidates `run_metrics`. Keep view DDL in one idempotent module and
re-run it after every migration. SQLite does **not** validate view bodies at creation time, so a view
referencing a dropped column stays in the schema and fails only at `SELECT` — add a smoke test that
`SELECT * FROM <each view> LIMIT 1` succeeds post-migration.

**Phase.** 3 (EVAL-01), 1 (the smoke test in CI).

---

### 7.10 Operational defaults: foreign keys OFF, journal DELETE, `check_same_thread` True

**Symptom.** `ON DELETE CASCADE` never fires and deleting a run orphans its rating. `database is
locked` while a slow CV method holds a write. `sqlite3.ProgrammingError: SQLite objects created in a
thread can only be used in that same thread`, intermittently and only for some endpoints.

**Cause.** Three separate defaults, all verified **[E]** and documented **[D]**:
- `PRAGMA foreign_keys` → **0**. OFF by default since 3.6.19, for backwards compatibility, and it is
  **connection-scoped** — not stored in the file.
- `PRAGMA journal_mode` → **`delete`**. A writer blocks all readers. WAL is the mode where *"readers
  do not block writers and a writer does not block readers"* **[D]**.
- `sqlite3.connect` defaults `check_same_thread=True`, and `sqlite3.threadsafety == 1` on this
  build **[E]** ("threads may share the module, but not connections"). FastAPI runs **non-async
  `def` endpoints in an anyio worker threadpool**, so a module-level connection created at import
  time gets touched from arbitrary worker threads.
- `PRAGMA busy_timeout` → **0** in raw SQLite but **5000 ms** via Python's `connect(timeout=5.0)`
  default **[E]**.

**Prevention.** One factory, every connection through it:

```python
def connect(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, check_same_thread=False, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")     # per-connection, every time
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA journal_mode=WAL")    # persists in the file
    conn.execute("PRAGMA synchronous=NORMAL")
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    return conn
```

and **one connection per request** via a FastAPI dependency, so no connection is ever shared across
threads:

```python
def get_db() -> Iterator[sqlite3.Connection]:
    conn = connect(DB_PATH)
    try:
        yield conn
    finally:
        conn.close()
```

Add a test that deletes a run and asserts the rating row is gone — the only way to catch a silently
disabled cascade. Use `with conn:` at write sites (commits on success, rolls back on exception) but
remember it does **not** open a transaction for DDL (§7.9). Use `BEGIN IMMEDIATE` for read-then-write
sequences: the busy handler is *not* invoked when a deferred read transaction tries to upgrade, so
`BEGIN IMMEDIATE` converts "SQLITE_BUSY at COMMIT" into "fail fast at BEGIN", which composes with
`busy_timeout`.

Two WAL caveats that bite a local single-user app specifically **[D]**: *"WAL does not work over a
network filesystem"* — so keep the DB out of NFS/SMB shares **and out of Dropbox/iCloud/Google Drive
sync folders**, a very live risk for a DB in the user's home directory. And the `-wal`/`-shm` sidecars
must travel with the file: `cp foo.db` while the app runs can lose committed data; use
`VACUUM INTO 'backup.db'` or the backup API.

**Phase.** 3.

---

## 8. Statistics: Wilson interval and Bradley-Terry

### 8.1 The Wald interval reports zero uncertainty at 0/n and n/n — and it is statsmodels' default

**Symptom.** A method with 3 thumbs-up from 3 runs shows "100% ± 0%". A method with 0/10 shows
"0% ± 0%". EVAL-14 exists to prevent exactly this and the default library call reintroduces it.

**Cause.** The Wald interval is `p̂ ± z·√(p̂(1−p̂)/n)`, which collapses to zero width at
`p̂ ∈ {0, 1}` *"falsely implying certainty"* **[D]**, and overshoots `[0, 1]` at small n. Measured
with `z = 1.9599639845400534` **[E]**:

| data | Wald | Wilson |
|---|---|---|
| 0/10 | **(0.0, 0.0)** | (0.00000, 0.27753) |
| 7/10 | (0.41597, **0.98403**) | (0.39678, 0.89221) |
| 1/3 | (**−0.20010**, 0.86677) | (0.06149, 0.79234) |

And `statsmodels.stats.proportion.proportion_confint`'s **default `method` is `'normal'`, i.e.
Wald** **[D]** — you must pass `method='wilson'` explicitly.

**Prevention.** See 8.2 for a dependency-free implementation. Never implement or default to Wald.

**Phase.** 3 (`/stats`), 8 (charts).

---

### 8.2 Wilson: exact formula, exact z, and the three edge cases

**Formula** (count form; `n_s` successes, `n_f = n − n_s`) **[D]**:

```
        n_s + ½z²  ±  z·√(n_s·n_f/n + z²/4)
w∓  =  ─────────────────────────────────────
                   n + z²
```

**Exact z** — compute it, don't hardcode `1.96`. Verified via `statistics.NormalDist().inv_cdf` **[E]**:
95% → `1.9599639845400534`, 90% → `1.6448536269514715`, 99% → `2.5758293035489`.

**Edge cases**, each with a closed form, all verified numerically **[E]**:
- **`p̂ = 0`** (`n_s = 0`): the `±` term is exactly `z²/2`, cancelling the `½z²` in the numerator.
  Interval is **`[0, z²/(n + z²)]`** — lower bound exactly 0, upper bound strictly positive.
  0/10 → `[0, 0.2775327998628891]`; 0/3 → upper `0.5614970317550453`.
- **`p̂ = 1`** (`n_f = 0`): interval is **`[n/(n + z²), 1]`**. 10/10 → `[0.722467200137111, 1.0]`.
- **`n = 0`**: `p̂` is undefined; the formula degenerates to `[0, 1]`. **Do not render `[0,1]` as an
  estimate** — it is "no information", and this is the same not-assessed-vs-assessed-zero distinction
  as EVAL-17, one layer up.

```python
import math
from statistics import NormalDist

def wilson_interval(successes: int, n: int, confidence: float = 0.95
                    ) -> tuple[float, float] | None:
    """Wilson score interval. None when n == 0 -- the caller MUST render that
    distinctly from (0.0, 1.0)."""
    if n < 0 or not (0 <= successes <= n):
        raise ValueError(f"invalid counts: {successes}/{n}")
    if n == 0:
        return None
    z = NormalDist().inv_cdf(1 - (1 - confidence) / 2)
    z2 = z * z
    denom = n + z2
    centre = (successes + z2 / 2) / denom
    half = (z / denom) * math.sqrt(successes * (n - successes) / n + z2 / 4)
    lo = 0.0 if successes == 0 else max(0.0, centre - half)
    hi = 1.0 if successes == n else min(1.0, centre + half)
    return (lo, hi)
```

**Three prevention notes beyond the formula:**
1. Watch for **`-0.0`**: 0/3 produces `-0.0` before clamping **[E]**, which serialises into JSON as
   `-0.0` and looks like a bug. The `max(0.0, ...)` handles it.
2. **Feed Wilson only non-NULL thumbs judgments.** `n = COUNT(thumbs_up)`, **not** `COUNT(*)` — the
   latter counts unrated runs as thumbs-down, which is §7.2's error resurfacing in the statistics
   layer:
   ```sql
   SELECT method,
          SUM(CASE WHEN thumbs_up = 1 THEN 1 ELSE 0 END) AS n_up,
          COUNT(thumbs_up)                               AS n_rated,
          COUNT(*)                                       AS n_runs
   FROM runs r LEFT JOIN ratings ra ON ra.run_id = r.id GROUP BY method;
   ```
3. **Rank by the lower bound, not by `p̂`** — that is the entire point of computing an interval, and it
   is what makes EVAL-14's "a rate from 4 ratings must not render like a rate from 400" actually
   operative. `1/1` (Wilson lower `0.2065`) then correctly ranks *below* `50/100` (lower `0.4038`),
   whereas `p̂` ranks it first.

**Phase.** 3 (compute + API), 4 (render), 8 (charts).

---

### 8.3 Bradley-Terry: a method that never loses gets an infinite score, and the fit reports success

**Symptom.** After 6 paired comparisons (EVAL-05/EVAL-15), method D shows strength `+18.7`, or `inf`,
or a number that grows with every iteration cap you raise. The optimizer reports convergence.
Rankings are nonsense and move on every refit.

**Cause.** BT models `Pr(i > j) = e^{β_i} / (e^{β_i} + e^{β_j})` **[D]**. The MLE is finite **iff** the
directed win-graph is **strongly connected** (Ford 1957 / Zermelo 1929): in every partition of the
methods into two non-empty sets, some method in the second beat some method in the first. If a method
never loses there is no path *out* of its node, the likelihood is monotonically increasing in `β_i`,
and `β_i → ∞`. An iterative solver just marches until its tolerance stops it — hence "converged".

With four methods and tens of human comparisons, a strongly-connected win graph is the **exception**,
not the rule. This is the expected state of Phase 8, not an edge case.

**Prevention.** Three layers, all of them:

1. **Check the condition before fitting and report it.** `n` is 4; this is cheap:
```python
def strongly_connected(wins: dict[tuple[str, str], float], methods: list[str]) -> bool:
    adj = {m: {j for j in methods if wins.get((m, j), 0) > 0} for m in methods}
    def reach(start: str) -> set[str]:
        seen, stack = {start}, [start]
        while stack:
            for nxt in sorted(adj[stack.pop()] - seen):   # sorted: see 6.3
                seen.add(nxt); stack.append(nxt)
        return seen
    return all(reach(m) == set(methods) for m in methods)
```
   If `False`, refuse to report absolute strengths, or flag the ranking as regularised-only. Always
   surface the per-pair comparison count next to it.

2. **Regularise with pseudo-games.** Add `ε` wins *and* `ε` losses to every ordered pair. This
   injects mild evidence of equality into every matchup, guarantees strong connectivity, and shrinks
   strength differences toward zero:
   ```python
   EPS = 0.5    # config knob, reported in the API response -- a convention, not a verified value
   w = {(i, j): wins.get((i, j), 0.0) + EPS
        for i in methods for j in methods if i != j}
   ```
   BT is identified only up to an additive constant on `β`, so **pin the scale** — normalise to
   geometric mean 1 on the `p`, or fix one method at `β = 0`. Forgetting this makes successive fits
   incomparable even when the ranking is stable.

3. **Store ties as a distinct outcome.** A human comparing two CV overlays will very often say
   "same". Counting a tie as half a win to each side is the pragmatic choice and helpfully contributes
   to strong connectivity; Davidson (1970) is the principled model
   (`P(tie) = δ√(λ_i λ_j) / (λ_i + λ_j + δ√(λ_i λ_j))`, with `δ = 0` recovering BT). Either way the
   database must keep the distinction so you can change your mind:
   ```sql
   CREATE TABLE comparisons (
     id INTEGER PRIMARY KEY,
     run_a_id INTEGER NOT NULL REFERENCES runs(id),
     run_b_id INTEGER NOT NULL REFERENCES runs(id),
     winner TEXT NOT NULL CHECK (winner IN ('a','b','tie')),
     created_at TEXT NOT NULL,
     CHECK (run_a_id <> run_b_id)
   );
   ```
   Note `winner` is `NOT NULL` **on purpose**: unlike the count fields, "no judgment" here is the
   *absence of a row*, so this CHECK genuinely fires (§7.1).

**Elo as the escape hatch.** Elo is BT with a fixed logistic scale and online updates
(`Pr(i > j) = 1/(1 + 10^{(R_j − R_i)/400})`) **[D]**. It never diverges and needs no connectivity
condition, but it is order- and path-dependent and yields no likelihood-based uncertainty. For an
offline harness that refits from scratch, regularised BT is better; Elo is the fallback if fitting
keeps causing trouble. The brief's EVAL-15 already allows either.

**Phase.** 8 (fit + report), 3 (the `comparisons` schema, so Phase 8 has data).

---

## 9. Canvas box drawing in the browser

> Empirically verified in Chromium 150 / Electron 43 at `devicePixelRatio === 2`, plus the CSSOM-View
> and WHATWG HTML specs. **This is the highest-risk area in the whole project for the reproducibility
> constraint** — a ±1 image-pixel error in the exemplar box changes the crop and therefore every
> downstream number, and no amount of backend determinism can recover it.

### 9.1 Three different "sizes", and the CSS→bitmap scale is **not** uniform across axes

**Symptom.** The drawn box appears where the user clicked, but the coordinates sent to the server are
wrong by a constant multiplicative factor — and the horizontal error differs from the vertical one.

**Cause.** A canvas has **three** independent sizes:

| size | how to read it | units | controls |
|---|---|---|---|
| **backing store** | `canvas.width` / `canvas.height` *attributes* | bitmap px | drawing resolution |
| **CSS layout** | `getBoundingClientRect()`, `style.width` | **CSS px** | apparent size; where the mouse is |
| **image natural** | `img.naturalWidth` / `naturalHeight` | image px | **the only space the server sees** |

The WHATWG HTML spec **[D]**: `width` defaults to **300** and `height` to **150**, and *"A `canvas`
element can be sized arbitrarily by a style sheet, its bitmap is then subject to the `object-fit` CSS
property"* — whose default for canvas is `fill`, i.e. **independent stretch in X and Y**.

So the two scale factors genuinely differ. Measured: `width=1000 height=600` with
`style="width:500.5px;height:300.25px"` gives **[E]**

```
rect   = {left: 20.5, top: 827.5, width: 500.5, height: 300.25}
scaleX = 1000/500.5   = 1.998001998001998
scaleY =  600/300.25  = 1.9983347210657785     // scaleX !== scaleY
```

Anyone who computes a single `scale`, or assumes it equals `devicePixelRatio`, is wrong the moment the
CSS box's aspect ratio drifts from the bitmap's — which any `width: 100%` responsive layout does.

**Prevention.** Never let CSS size and attribute size drift; derive one from the other in one place,
driven by a `ResizeObserver`. Note the equality guard — **assigning to `canvas.width` resets the
bitmap and the context state (including the transform)** even when the value is unchanged:

```js
function syncCanvasSize(canvas) {
  const dpr  = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();          // CSS px, fractional
  const bw = Math.round(rect.width * dpr), bh = Math.round(rect.height * dpr);
  if (canvas.width !== bw || canvas.height !== bh) {     // guard: assignment clears the bitmap
    canvas.width = bw; canvas.height = bh;
  }
  canvas.style.width  = rect.width  + 'px';
  canvas.style.height = rect.height + 'px';
  return { cssW: rect.width, cssH: rect.height, dpr };
}
```

**Phase.** 4.

---

### 9.2 The full transform chain, its exact inverse, and where `dpr` belongs

**Symptom.** Drawn boxes are right but server-returned result boxes are misaligned, or vice versa —
and the two directions drift apart as soon as zoom or pan is used.

**Prevention.** Model the viewport as one affine in **CSS-pixel space** and write both directions
adjacent to each other:

```js
// view = { dpr, zoom, panX, panY }   zoom = CSS px per image px; pan in CSS px
function toImagePoint(ev, canvas, view) {
  const r = canvas.getBoundingClientRect();   // re-measure EVERY event; CSS px (spec)
  const cssX = ev.clientX - r.left;           // NOT offsetX (9.3), NOT offsetLeft (9.4)
  const cssY = ev.clientY - r.top;
  return { x: (cssX - view.panX) / view.zoom,   // IMAGE space, float
           y: (cssY - view.panY) / view.zoom };
}
function toCssPoint(pt, view) {               // exact inverse
  return { x: pt.x * view.zoom + view.panX,
           y: pt.y * view.zoom + view.panY };
}
```

**`dpr` does not appear in the pointer math** — that is the key simplification. `getBoundingClientRect`
and `clientX` are both CSS px (the CSSOM-View spec: *"All coordinates and dimensions for the APIs
defined in this specification are in CSS pixels, unless otherwise specified"* **[D]**; confirmed
empirically by `rect.left === 20.5` at DPR 2 **[E]**), and `canvas.width = cssW * dpr` from 9.1 means
the drawing side absorbs `dpr` exactly once:

```js
const k = view.dpr * view.zoom;
ctx.setTransform(k, 0, 0, k, view.panX * view.dpr, view.panY * view.dpr);
ctx.drawImage(img, 0, 0);              // now the context IS image space
ctx.lineWidth = 1 / k;                 // keep hairlines 1 device px
ctx.strokeRect(b.x0, b.y0, b.x1 - b.x0, b.y1 - b.y0);   // server integers, verbatim
```

The payoff is that **result boxes are stroked with the exact integers the server returned, with zero
client-side arithmetic** — which is what makes the overlay provably aligned with the boxes the rater
is judging (UI-03, UI-05).

**Do not rely on round-trip exactness.** `(x·z + p − p)/z` is not an IEEE-754 identity. So the box in
**image space is the single source of truth**: store it, send it, receive it, and never regenerate it
from screen coordinates.

**Test recipe.** For `zoom ∈ {0.5, 1, 2.37}`, `dpr ∈ {1, 2}` and several pans, assert
`toImagePoint(toCssPoint(p)) ≈ p`; then have the synthetic generator (EVAL-03) emit an image with a
1 px crosshair at a known pixel, draw a box at that image coordinate, and assert by screenshot that
they coincide. Caveat for the test itself: Chromium's `MouseEventInit` **coerces `clientX` to a
long** — passing `137.5` reads back `137` **[E]** — so synthetic-event tests cannot exercise the
fractional path. Use real Playwright input for that.

**Phase.** 4.

---

### 9.3 `event.offsetX/offsetY` is **rounded to an integer** and does not subtract padding

This is the most concrete threat to "same image + box ⇒ identical results" in the entire project.

**Symptom.** The same visual drag yields image coordinates differing by ±1 px between sessions or
browsers. At a 2× CSS→bitmap scale a half-CSS-pixel error becomes a **whole image pixel**, changing
the crop and therefore the results hash.

**Cause.** The CSSOM-View spec, verbatim **[D]**:

> "The **offsetX** attribute must follow these steps: If the event's dispatch flag is set, return the
> x-coordinate of the position where the event occurred relative to the origin of the **padding edge**
> of the target node, **ignoring the transforms that apply to the element and its ancestors**, and
> terminate these steps. Return the value of the event's **pageX** attribute."

Two spec-level landmines: the origin is the **padding edge** (not the content box), and if the
dispatch flag is not set `offsetX` **silently falls back to `pageX`** — an entirely different origin.

Measured in Chromium 150 **[E]**:

```
no border/padding, rect.left = 20.5
  clientX = 144  ->  clientX - rect.left = 123.5
                     event.offsetX       = 124        <-- ROUNDED; 0.5 CSS px lost

border:3px padding:5px, rect.left = 37.5
  clientX = 137  ->  clientX - rect.left - borderLeft = 96.5
                     event.offsetX                    = 97          <-- rounded again
                     (padding actually NOT subtracted: 91.5 is not what you get)

transform: scale(1.5) translate(10px,0);  rect.width = 300, offsetWidth = 200
  click at rect.left + 30  ->  event.offsetX === 20   (== 30/1.5)
```

So Chromium's real behaviour is `offsetX ≈ round(clientX − rect.left − borderLeftWidth)`, and under a
CSS transform `offsetX` divides the transform out while `clientX − rect.left` does not — the two
disagree by the transform scale.

**Prevention.** Never use `offsetX`/`offsetY` for anything that must be exact. Use `clientX/clientY`
minus a freshly measured rect (9.2), and eliminate the confounders by CSS contract:

```css
#stage {
  display: block;
  border: 0;            /* removes the offsetX/rect asymmetry */
  padding: 0;
  transform: none;      /* keep CSS transforms off the canvas itself */
  touch-action: none;   /* mandatory, else Pointer Events get eaten by scroll/zoom (9.5) */
}
```

**Related: never round intermediates.** `DOMRect` members are doubles — measured `rect.left = 37.5`,
`rect.top = 11.25`, `rect.width = 500.5` **[E]**. Any `Math.round(rect.left)` throws away up to
0.5 CSS px, i.e. up to ~1 image px at these scales. Keep the whole chain in floats and round **exactly
once**, in image space, in `finalizeBox` (9.4).

**Phase.** 4.

---

### 9.4 Negative-extent drags, clamping, and the exclusive edge — round once, at the end

**Symptom.** Dragging right-to-left or bottom-to-top produces a zero-area or negative box.
`ctx.strokeRect` renders negative extents happily, so it is invisible in the UI; the server either
422s or accepts it and `img[y0:y1, x0:x1]` silently yields a **zero-size array** (§1.5). Boxes drawn
partly off-image produce crops smaller than the box.

**Cause.** A drag gives two corners in arbitrary order, and NumPy slicing is **exclusive** on the
upper bound: `np.arange(100).reshape(10,10)[2:5, 3:7]` has shape `(3, 4)` **[E]**, i.e.
`h = y1 − y0`, `w = x1 − x0`, no `+1` anywhere. If the UI sends an *inclusive* `x1, y1`, every crop is
1 px short in each dimension — which drops the NCC self-match below 1.0 and makes every reported box
size off by one.

**Prevention.** Normalise → clamp in float → integerise → re-clamp, in exactly that order, once, at
`pointerup`:

```js
function finalizeBox(a, b, natW, natH, minSize = 8) {
  // 1) normalise: handles any drag direction
  let x0 = Math.min(a.x, b.x), x1 = Math.max(a.x, b.x);
  let y0 = Math.min(a.y, b.y), y1 = Math.max(a.y, b.y);
  // 2) clamp in FLOAT space, before rounding
  x0 = Math.max(0, Math.min(x0, natW)); x1 = Math.max(0, Math.min(x1, natW));
  y0 = Math.max(0, Math.min(y0, natH)); y1 = Math.max(0, Math.min(y1, natH));
  // 3) integerise: floor the inclusive start, ceil the exclusive end (grow outward)
  x0 = Math.floor(x0); y0 = Math.floor(y0);
  x1 = Math.ceil(x1);  y1 = Math.ceil(y1);
  // 4) re-clamp (ceil can overshoot) and reject rather than silently fix
  x1 = Math.min(x1, natW); y1 = Math.min(y1, natH);
  if (x1 - x0 < minSize || y1 - y0 < minSize) return null;
  return { x0, y0, x1, y1 };            // half-open [x0,x1) x [y0,y1)
}
```

`floor`/`ceil` beats `round` on both edges: `round` can shrink a thin selection to zero width, and its
behaviour at exactly `.5` is a tie-break you don't want load-bearing. Adopt half-open `[x0, x1)`
everywhere, write it into the `ExemplarBox` docstring (*"x1/y1 exclusive; width = x1 − x0"*), and make
the frozen Pydantic model enforce `x1 > x0`, `y1 > y0`, `x1 <= image_width`, `width >= 8`,
`height >= 8` — so a stray click is a 422, not an empty result.

**Phase.** 4 (UI), 1 (schema).

---

### 9.5 Losing `pointerup` outside the canvas, and `pointercancel` really does fire

**Symptom.** The user drags a box, releases over the results panel, and the app stays stuck in
"dragging" state; the next mouse move anywhere resizes a phantom box.

**Cause.** Mouse events target whatever element is under the cursor, so once the cursor leaves the
canvas neither `mousemove` nor `mouseup` fires on it.

**Prevention.** Pointer Events with `setPointerCapture`. MDN **[D]**: it *"designates a specific
element as the capture target of future pointer events. Subsequent events for the pointer will be
targeted at the capture element until capture is released"*, and capture releases automatically on
`pointerup` **and** `pointercancel`. Both APIs confirmed present **[E]**.

```js
let drag = null;
canvas.addEventListener('pointerdown', (e) => {
  if (e.button !== 0 || !e.isPrimary) return;
  canvas.setPointerCapture(e.pointerId);      // we now own move/up/cancel, everywhere
  drag = { id: e.pointerId, start: toImagePoint(e, canvas, view) };
  e.preventDefault();
});
canvas.addEventListener('pointermove', (e) => {
  if (!drag || e.pointerId !== drag.id) return;
  drag.current = toImagePoint(e, canvas, view);
  requestAnimationFrame(render);               // never draw synchronously in the handler
});
const end = (e) => {
  if (!drag || e.pointerId !== drag.id) return;
  const box = finalizeBox(drag.start, toImagePoint(e, canvas, view),
                          img.naturalWidth, img.naturalHeight);
  drag = null;                                 // reset state on BOTH paths
  if (box) submit(box);
};
canvas.addEventListener('pointerup', end);
canvas.addEventListener('pointercancel', end);         // OS gesture steals the pointer
canvas.addEventListener('lostpointercapture', (e) => {
  if (drag && e.pointerId === drag.id) drag = null;
});
```

`touch-action: none` (9.3) is mandatory or trackpad/mobile scroll-zoom will `pointercancel` your drag.
A handler that listens only for `pointerup` leaks state — `pointercancel` fires in the wild.

**Phase.** 4.

---

### 9.6 `setTransform` resets, `scale`/`translate` accumulate — and `offsetLeft` is useless

**Symptom.** After enough zoom/pan cycles the overlay drifts a few pixels off the image and only a
page reload fixes it. Or: the box lands correctly at scroll position 0 and drifts by exactly the
scroll offset once the page is scrolled.

**Cause.** Two separate confirmed behaviours.

`ctx.scale`/`ctx.translate` **multiply into** the current matrix; `setTransform` **replaces** it.
Measured **[E]**: `setTransform(1,0,0,1,0,0); scale(2,2); scale(2,2); translate(5,5)` →
`{a:4, d:4, e:20, f:20}` (2·2 = 4, and the translate is pre-multiplied by the scale: 5·4 = 20), while
a subsequent `setTransform(3,0,0,3,7,7)` → `{a:3, d:3, e:7, f:7}`. MDN, verbatim **[D]**:
`setTransform()` *"resets (overrides) the current transformation to the identity matrix, and then
invokes a transformation described by the arguments."*

And `offsetLeft`/`offsetTop` are relative to `offsetParent` — the nearest *positioned* ancestor,
frequently `<body>`. Measured **[E]**: both test canvases reported `offsetLeft === 0` and
`offsetTop === 0` while sitting at `rect.left = 20.5` and `rect.left = 15`. Useless.
`getBoundingClientRect()` is what you need because *"the amount of scrolling that has been done of the
viewport area (or any other scrollable element) is taken into account"* **[D]** — which is exactly
right, since `clientX/clientY` are also viewport-relative, so the scroll cancels. It also reflects CSS
transforms in current Chromium (`rect.width === 300` for `scale(1.5)` on a 200 px canvas, while
`offsetWidth` stays 200) **[E]** — note an MDN summary claiming otherwise is out of date; trust the
measurement.

**Prevention.** One absolute `setTransform` per frame, computed from the view-state object, never
accumulated (the snippet in 9.2). To draw in raw backing-store pixels — clearing, for instance —
reset explicitly:

```js
ctx.setTransform(1, 0, 0, 1, 0, 0);
ctx.clearRect(0, 0, canvas.width, canvas.height);
```

And re-measure `getBoundingClientRect()` **inside** every event handler; never cache it across
`pointermove`, because layout, scroll, or a sidebar toggle can move the canvas mid-drag.

**Phase.** 4.

---

### 9.7 `object-fit: contain` letterboxing creates dead margins that map nowhere

**Symptom.** Clicks map correctly near the centre and get progressively wrong toward the edges; clicks
in the top/bottom bands map to negative or out-of-range image coordinates.

**Cause.** Measured **[E]**: an `<img>` with `naturalWidth/Height = 100 × 70` in a 400 × 400 box with
`object-fit: contain` gives `scale = min(4, 5.71) = 4`, drawn size 400 × 280, and
`deadMarginY = (400 − 280)/2 = 60` CSS px of unmapped space **top and bottom**.
`getBoundingClientRect()` on the `<img>` still reports 400 × 400 — nothing tells you about the 120 px
of dead space.

**Prevention.** Do not display the image in an `<img>` with `object-fit`. Draw it into the canvas with
`ctx.drawImage` and own the fit yourself, so the letterbox **is** the pan:

```js
function fitContain(natW, natH, cssW, cssH) {
  const zoom = Math.min(cssW / natW, cssH / natH);
  return { zoom, panX: (cssW - natW * zoom) / 2, panY: (cssH - natH * zoom) / 2 };
}
```

Then the single inverse formula from 9.2 handles letterboxing for free. This is the same class of bug
as FastSAM's letterbox round-trip (§5.3), one layer up — and the same fix applies: one function owns
the transform, and its inverse is written next to it.

**Phase.** 4.

---

### 9.8 DPR changes mid-session, and `device-pixel-content-box` cannot be relied on

**Symptom.** The user drags the window from a Retina display to a 1× monitor (or hits Cmd-+). The
canvas goes blurry or pixel-doubled, **and box coordinates shift**, because the backing store no
longer matches the CSS box.

**Cause.** `devicePixelRatio` changes and nothing fires a `resize` that a naive app listens for. A
`resolution` media query is pinned to the value it was created with, so it must be **re-created after
every change** **[D]**.

**Prevention.**

```js
let off = null;
function watchDpr(onChange) {
  const relisten = () => {
    off?.();
    const mql = matchMedia(`(resolution: ${window.devicePixelRatio}dppx)`);
    mql.addEventListener('change', relisten);          // fires when DPR leaves this value
    off = () => mql.removeEventListener('change', relisten);
    onChange(window.devicePixelRatio);
  };
  relisten();
}
watchDpr(() => { syncCanvasSize(canvas); render(); });
new ResizeObserver(() => { syncCanvasSize(canvas); render(); }).observe(canvas);
```

The query form is confirmed correct (`matchMedia('(resolution: 2dppx)').matches === true` at DPR 2
**[E]**).

**Do not depend on `device-pixel-content-box`.** MDN documents it as *"the size of the content area as
defined in CSS, in device pixels, before applying any CSS transforms"* **[D]** — theoretically the
right signal for canvas sizing — but `ro.observe(canvas, {box: 'device-pixel-content-box'})` **never
fired** in Chromium 150 / Electron 43 over a 1.5 s window while the default box worked **[E]**.
Feature-detect it; keep `rect × dpr` as the load-bearing path.

**Phase.** 4.

---

## 10. Cross-cutting traps the brief misses

### 10.1 EXIF orientation: the browser rotates, OpenCV rotates, PIL does not

**Symptom.** For a photo taken on a phone, the box the user draws is 90° away from where the backend
looks. Only some uploaded images are affected (API-06).

**Cause.** JPEG EXIF carries an orientation tag. Browsers apply it when rendering `<img>`.
`cv2.imread` applies it too by default (and `IMREAD_IGNORE_ORIENTATION` disables that). PIL/Pillow's
`Image.open` does **not** apply it unless you call `ImageOps.exif_transpose`. So a pipeline that
displays via the browser, reads via OpenCV, and thumbnails via Pillow has **two** different
orientations in play. (See §Unverified — the OpenCV EXIF default was not re-tested here.)

**Prevention.** Strip the ambiguity at ingest. On upload, decode once, apply orientation once,
re-encode to a canonical **PNG** (no EXIF) under a content-addressed name, and serve *that* file to
both the frontend and the methods. The `images` table stores the canonical path and its sha256. Then
"same image" in the reproducibility constraint is a hash, not a filename, and no layer can disagree
about orientation.

**Phase.** 3 (API-06 upload), 1 (demo asset preparation — do the same to the basketball frames).

---

### 10.2 The exemplar's self-match is a Method 1 problem too, and it inflates every score

**Symptom.** Every method reports at least one perfect match. Precision looks great on the runs where
nothing else was found.

**Cause.** IDEA.md §5 (2c) raises this for Method 2 only. But because search is confined to one image
(§4), *every* method finds the exemplar: NCC scores exactly `1.0` at the exemplar location **[E]**,
`dino-dense` peaks there by construction, and FastSAM almost certainly proposes it. If it is silently
included, `R` is inflated by 1 in every run and the human is asked to rate a match they drew
themselves.

**Prevention.** Make it a first-class field of `Match`, not a convention:

```python
class Match(BaseModel, frozen=True):
    box: Box
    score: float
    is_exemplar: bool = False       # True iff IoU with the exemplar box > 0.5
```

Set it in `common/nms.py` (one place, all methods) and report **both** counts in `SearchResult`:
`returned_count` and `returned_count_excl_exemplar`. Decide once, in the store, which one `R` means
for the EVAL-07 metric views, write it into the duplicate/fragment convention doc that EVAL-16 already
requires the UI to display, and grey the exemplar box out in the overlay so the rater is never asked
to judge it.

**Phase.** 2 (`nms.py` + `Match` schema), 3 (which count `R` uses), 4 (rendering).

---

### 10.3 `SearchMethod` is a `Protocol`, so MyPy strict will not check the registry unless you make it

**Symptom.** A new method with a subtly wrong `search` signature (say `exemplar: Box` instead of
`ExemplarBox`) registers fine and fails at runtime. INFRA-10's "adding a method touches exactly one
new file plus one import" quietly becomes "one file plus a debugging session".

**Cause.** A decorator registry typed as `dict[str, Any]` erases the protocol. `Protocol` conformance
is only checked where a value is *assigned to* a variable of the protocol type — a decorator that
takes `Callable[..., Any]` never triggers that check.

**Prevention.** Type the decorator so registration *is* the assignment:

```python
_REGISTRY: dict[str, SearchMethod] = {}     # dict, not set -- ordered (see 6.3)

M = TypeVar("M", bound=SearchMethod)

def register_method(cls: type[M]) -> type[M]:
    inst: SearchMethod = cls()              # <- the assignment MyPy checks
    if inst.name in _REGISTRY:
        raise ValueError(f"duplicate method name: {inst.name}")
    _REGISTRY[inst.name] = inst
    return cls
```

Add a parametrised test over `_REGISTRY.values()` asserting each method's `config_model` is a frozen
`BaseModel` with a JSON schema (API-01 depends on that), that instantiating it with no arguments
succeeds (so the UI form has defaults), and that `search` returns a `SearchResult` on a 64×64
synthetic image. That test is the actual guarantee behind INFRA-10, and it makes the duplicate-name
collision a test failure rather than a silent overwrite.

**Phase.** 1.

---

### 10.4 Latency measured with `time.time()` is not a latency breakdown

**Symptom.** EVAL-11's preprocess/inference/postprocess numbers don't sum to the total, or go
backwards, or show 0.0 ms for fast stages.

**Cause.** `time.time()` is wall-clock and subject to NTP adjustment; it can go backwards.
`time.perf_counter()` is monotonic and is the right primitive. Separately, the first ONNX Runtime
`run()` on a session includes lazy allocation and is not representative — so the first run of every
benchmark sweep is an outlier that shifts the percentiles UI-06 reports.

**Prevention.** `time.perf_counter_ns()` for every stage, a typed `LatencyBreakdown` frozen model with
an explicit `other_ms = total - (pre + infer + post)` field so the arithmetic is visible rather than
assumed, and a warm-up run in the benchmark runner (EVAL-04) that is executed and discarded. Since
sessions are created once at startup (API-07), warm up in `lifespan` too, so the first user request is
not the outlier.

**Phase.** 3 (API-07 + EVAL-11), 8 (benchmark warm-up).

---

### 10.5 The threshold is a per-run datum, not a config datum

**Symptom.** The EVAL-08 offline threshold sweep can't be reconstructed, because the stored config
records `calibration_strategy: "gmm"` but not the number it produced.

**Cause.** All three calibration strategies (`self-similarity`, `ratio`, `gmm`) compute the threshold
*from the image*. It is an **output**, not an input. Storing only the strategy name means you know how
the threshold was chosen but not what it was — and the sweep needs the applied value to place the
operating point on the curve. EVAL-08 says "with raw scores *and* the applied threshold"; the trap is
that it's easy to satisfy that for a fixed-threshold method and forget it for the calibrated ones.

**Prevention.** `applied_threshold: float` is a required field of `SearchResult`, persisted per run
alongside the candidate list — plus `threshold_strategy` and, for `gmm`, the fitted parameters in
`diagnostics` (a two-component GMM fit is itself a stochastic step needing a seed; `sklearn`'s
`GaussianMixture` takes `random_state`, and the brief's §8 reproducibility constraint applies). Also
guard the `gmm` failure mode the brief doesn't mention: on a unimodal similarity histogram the
two-component fit converges to two overlapping components and the "cut between modes" is arbitrary.
Detect it (component means closer than one pooled std) and fall back to `self-similarity` with a
recorded `threshold_fallback_reason`.

**Phase.** 2 (`calibration.py` + `SearchResult` field), 3 (persistence).

---

## Sources

Fetched and read directly:

- https://docs.opencv.org/4.x/df/dfb/group__imgproc__object.html — `matchTemplate` formulas, result size `(W−w+1)×(H−h+1)`, mask semantics
- https://raw.githubusercontent.com/opencv/opencv/4.10.0/modules/calib3d/src/ptsetreg.cpp — `RNG rng((uint64)-1)` at lines 171 and 284, `modelPoints=2`, `confidence=0.99`, `maxIters=1000`
- https://github.com/opencv/opencv/blob/master/modules/core/src/rand.cpp — `theRNG()` is thread-local (`getCoreTlsData().rng`)
- https://raw.githubusercontent.com/opencv/opencv/4.x/modules/calib3d/src/usac/ransac_solvers.cpp — `random_generator_state = 0`, `isParallel = false`, `MAX_THREADS = getNumThreads()`
- https://raw.githubusercontent.com/opencv/opencv/4.x/modules/imgproc/src/templmatch.cpp — DFT `blocksize`/`dftsize`/`tileCount` derived from the correlation-map size
- https://github.com/opencv/opencv/issues/24933 — maintainer `vpisarev`: "RANSAC should always return the same subset"
- https://github.com/opencv/opencv/issues/24835 — closed "won't fix" on the reproducibility principle
- https://github.com/opencv/opencv/issues/15277, https://github.com/opencv/opencv/issues/9694 — `setNumThreads` not honoured
- https://docs.opencv.org/4.10.0/db/d95/classcv_1_1ORB.html — `ORB::create` defaults
- https://github.com/opencv/opencv/issues/5688 — flat-template `TM_CCOEFF_NORMED` returns 1 (OpenCV 4.x)
- https://github.com/opencv/opencv/issues/20557 — NaN with masked `TM_CCOEFF_NORMED`
- https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf — Lowe IJCV 2004, §7.3 bin sizes and hash table, §7.4 ≥3 entries, half-error-range outlier rejection, 0.98 acceptance
- https://github.com/facebookresearch/dinov2/blob/main/dinov2/models/vision_transformer.py — token layout, `interpolate_offset=0.1`, bicubic + antialias
- https://raw.githubusercontent.com/facebookresearch/dinov2/main/dinov2/data/transforms.py — ImageNet mean/std, eval transform
- https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/dinov2/modeling_dinov2.py — no spatial-size validation, bicubic + `align_corners=False`, no 0.1 offset
- https://github.com/huggingface/transformers/issues/37817 — register tokens included in patch tokens (the shift bug, in a shipped library)
- https://github.com/huggingface/transformers/issues/34292 — DINOv2 patch size is 14, documented as 16
- https://raw.githubusercontent.com/ultralytics/ultralytics/main/examples/YOLOv8-Segmentation-ONNXRuntime-Python/main.py — `process_mask`: matmul → upsample → crop → `gt_(0.0)`
- https://docs.ultralytics.com/models/fast-sam/ — `imgsz=1024, conf=0.4, iou=0.9, retina_masks=True`
- https://arxiv.org/pdf/2306.12156 — FastSAM: over-segmentation from dense point grids
- https://onnxruntime.ai/docs/performance/tune-performance/threading.html — `intra_op_num_threads` / `inter_op_num_threads` / `execution_mode` defaults; **no** determinism statement
- https://onnxruntime.ai/docs/api/python/api_summary.html — `SessionOptions` attribute names
- https://raw.githubusercontent.com/microsoft/onnxruntime/main/onnxruntime/core/mlas/lib/sgemm.cpp — GEMM partitions M/N only, never K; K-blocking is a function of K alone
- https://raw.githubusercontent.com/microsoft/onnxruntime/main/onnxruntime/core/providers/cpu/reduction/reduction_ops.cc — reduction algorithm selected via `ThreadPool::DegreeOfParallelism` (the latent thread-count branch)
- https://raw.githubusercontent.com/microsoft/onnxruntime/main/onnxruntime/core/framework/session_options.h — `use_deterministic_compute` comment and default
- https://github.com/microsoft/onnxruntime/issues/18672 — consecutive runs differ on CPU EP; fixed by disabling `enable_mem_pattern`
- https://github.com/microsoft/onnxruntime/issues/28018 — non-deterministic TopK with default `SessionOptions`; "always explicitly create SessionOptions"
- https://github.com/microsoft/onnxruntime/issues/23335 — cross-binding output divergence, no maintainer explanation
- GitHub code search `use_deterministic_compute repo:microsoft/onnxruntime` (23 hits) — every consumer is CUDA or training; none in `providers/cpu/*`
- https://sqlite.org/lang_aggfunc.html — `avg`/`sum`/`total`/`count` NULL semantics
- https://sqlite.org/nulls.html — `GROUP BY` NULL grouping, "somewhat arbitrary"
- https://sqlite.org/lang_createtable.html — CHECK-passes-on-NULL, UNIQUE-NULLs-distinct
- https://sqlite.org/lang_altertable.html — the four supported operations, ADD COLUMN restrictions, the 12-step rebuild
- https://sqlite.org/pragma.html — `user_version`, `foreign_keys` default OFF, `journal_mode` default DELETE
- https://sqlite.org/json1.html — `->`/`->>`, JSONB in 3.45.0, treat as opaque
- https://sqlite.org/gencol.html — generated columns 3.31.0; STORED not addable via ALTER TABLE
- https://sqlite.org/limits.html — `SQLITE_MAX_LENGTH` default 1,000,000,000
- https://sqlite.org/fileformat2.html — page payload `X = U − 35` = 4061 at 4096-byte pages
- https://sqlite.org/wal.html — readers/writers don't block; WAL does not work over network filesystems
- https://docs.python.org/3/library/sqlite3.html — `check_same_thread`, `isolation_level`, `timeout`, `threadsafety`, `autocommit`
- https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval — Wilson formula, Wald's zero-width failure, continuity correction
- https://www.statsmodels.org/stable/generated/statsmodels.stats.proportion.proportion_confint.html — default `method='normal'` (Wald)
- https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model — BT probability, MM update, Elo correspondence
- https://drafts.csswg.org/cssom-view/ — §10 `offsetX` padding-edge + `pageX` fallback + transform-ignoring; "all coordinates … in CSS pixels"
- https://html.spec.whatwg.org/multipage/canvas.html — canvas default 300×150; CSS sizing subject to `object-fit`
- https://developer.mozilla.org/en-US/docs/Web/API/CanvasRenderingContext2D/setTransform — "resets (overrides) the current transformation"
- https://developer.mozilla.org/en-US/docs/Web/API/Element/getBoundingClientRect — scrolling is taken into account
- https://developer.mozilla.org/en-US/docs/Web/API/MouseEvent/offsetX
- https://developer.mozilla.org/en-US/docs/Web/API/Window/devicePixelRatio — the HiDPI recipe; re-creating the `resolution` media query
- https://developer.mozilla.org/en-US/docs/Web/API/Canvas_API/Tutorial/Optimizing_canvas
- https://developer.mozilla.org/en-US/docs/Web/API/Element/setPointerCapture — capture released on `pointerup`/`pointercancel`
- https://developer.mozilla.org/en-US/docs/Web/API/ResizeObserver/observe — `device-pixel-content-box`
- https://numpy.org/doc/stable/reference/generated/numpy.sum.html — pairwise summation depends on axis and memory layout
- https://numpy.org/doc/stable/reference/generated/numpy.argmax.html — first occurrence on ties
- https://docs.python.org/3/using/cmdline.html#envvar-PYTHONHASHSEED

Local empirical verification (all `[E]` claims), across two environments: OpenCV 4.10.0 and 5.0.0
(macOS wheels, `Parallel framework: GCD`), NumPy 1.26 / 2.0 / 2.5.1 (Accelerate BLAS), onnxruntime
1.19.2 / 1.27.0 (CPU EP), onnx, SQLite 3.51.0, Python 3.9.6 / 3.12 / 3.13, macOS arm64
(Darwin 25.5.0); browser probes in Chromium 150 / Electron 43 at `devicePixelRatio === 2`.

---

## Unverified

Flagged so nothing here is mistaken for a measured fact.

1. **`estimateAffinePartial2D`'s Python-level `maxIters` default.** The C++ registrator default is
   `1000` **[D]**; the Python binding's own `maxIters` parameter default (commonly quoted as 2000) was
   not read from the signature. Pass it explicitly and record it in config.
2. **Whether ONNX Runtime CPU determinism across thread counts generalises.** Measured `diff = 0.0`
   for Conv→ReduceMean→MatMul (ORT 1.19.2) and MatMul→ReduceSum→ReduceSum→Softmax and Conv+Relu+Conv
   (ORT 1.27.0), both macOS arm64 **[E]**, with a structural argument from MLAS's M/N-only
   partitioning **[D]**. Not a general guarantee: `reduction_ops.cc` *does* select different code
   paths by thread count **[D]**, and those two paths merely happened to agree bitwise here. Verify
   with a same-input-twice assertion on the *actual* DINOv2 and FastSAM graphs in Phase 6/7.
3. **YOLOv8 `max_det = 300` default.** Corroborated from secondary sources describing "300 bounding
   boxes per image"; not read from the Ultralytics config reference. Read `default.yaml` before
   relying on the number.
4. **Whether Ultralytics `process_mask` (crop-in-proto-space) vs `process_mask_upsample` /
   `process_mask_native` differ in order as described.** The upsample→crop order was read verbatim
   from the ONNX Runtime example **[D]**; the existence and behaviour of the cheaper crop-first
   variant is from memory of the Ultralytics `ops` module and was not fetched.
5. **`cv2.imread`'s EXIF-orientation default** (§10.1). Not re-tested. The mitigation (canonicalise to
   PNG at ingest) is correct regardless of the answer, which is why it is the recommendation.
6. **Whether masks were supported for `TM_CCOEFF_NORMED` in OpenCV versions older than 4.10.** Verified
   working in 4.10.0 and 5.0.0 **[E]**; the historical restriction to `TM_SQDIFF`/`TM_CCORR_NORMED` and
   the version it was lifted in were not established. Pin ≥ 4.10 if you use masked NCC.
7. **`TEMPLATE_STD_FLOOR = 2.0`** (§1.1) and **`min_members = 3` / vote-weight floors** (§2.3) are
   starting points, not measured optima. Sweep them on the EVAL-03 synthetic set in Phase 8.
8. **The claim that inline JSON blobs measurably slow narrow scans** (§7.8). The 4061-byte overflow
   threshold is verified **[D]**; the performance *magnitude* was not benchmarked. The table split is
   cheap insurance either way.
9. **Ford (1957) / Zermelo (1929) strong connectivity as necessary *and* sufficient for a finite BT
   MLE** (§8.3). Consistent across multiple secondary sources; the primary PDF was not readable.
   Re-check against Hunter (2004) before citing it in writing.
10. **Davidson (1970) tie formula** and **Rao–Kupper**. From secondary summaries, not the papers. The
    `δ = 0 ⇒ BT` reduction is a good self-check and holds.
11. **`EPS = 0.5` pseudo-games** (§8.3) is a convention analogous to a Jeffreys/Haldane adjustment,
    not a verified recommendation. Expose it as a config knob; don't bake it in.
12. **The equivalence of pseudo-games, phantom players, and specific Bayesian priors** as BT
    regularisers. That each one *fixes* separation is standard and safe; exact mathematical
    equivalence is unverified — do not assert it.
13. **Section 9's browser measurements are Chromium 150 / Electron 43 only**, at DPR 2. Firefox and
    WebKit were not tested; `offsetX` rounding and the transform interaction are spec-adjacent enough
    to be likely-universal, but the numbers are one engine's. Also unverified: whether real
    (non-synthetic) `MouseEvent.clientX` is ever fractional in Chromium — the recommendation to keep
    the chain in floats is correct regardless, since `rect.left` demonstrably is. Write the 9.2
    round-trip test in Phase 4 **before** the drawing code.
14. **`ResizeObserver` `device-pixel-content-box`** — established only that it did not fire in
    Chromium 150 / Electron 43 within 1.5 s while the default box worked **[E]**. Whether that is a
    bug, a platform gap, or a misuse was not determined. Feature-detect; don't make it load-bearing.
15. **BLAS thread-count invariance on x86/OpenBLAS.** Measured no effect on arm64/Accelerate **[E]**;
    OpenBLAS on x86-64 was not tested. Set the env vars anyway.
16. **`cv2.setNumThreads(1)` being ignored is a macOS/GCD-backend property.** Linux conda-forge builds
    typically use TBB or pthreads, where `1` may well work. Since the project pins conda-forge OpenCV,
    re-check `getNumThreads()` after the assertion on the actual CI image — and keep the assertion, it
    is what makes the difference visible.
17. **The continuity-corrected Wilson variant.** Transcribed and sanity-checked (uniformly wider,
    symmetric under `k → n−k`) but the plain interval is what EVAL-14 needs and is what is fully
    verified. Its discriminant can go negative for `n = 1` — clamp at 0 if you ship it.
18. **`PYTHONHASHSEED=0` in `pixi.toml`.** Suggested as belt-and-braces; not tested that pixi
    propagates it to every task invocation.
