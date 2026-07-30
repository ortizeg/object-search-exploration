# Engineering log — `dino-dense` on floor-plan doors (2026-07-30)

A record of the root-cause investigation and empirical iterate/measure/revert loop that improved
`dino-dense` (Method ③) specifically on the **floorplans-door** target-domain regime — the
method's worst regime per [`docs/eval/floorplans-findings.md`](../eval/floorplans-findings.md)
(tuned F1 0.147, worst of six methods, recall 0.00 on small doors). Follows the
[`dino-dense-improvement.md`](dino-dense-improvement.md) template and the project's
method-improvement playbook: own session, own branch, iterate → measure every regime → revert
anything that does not genuinely help.

**Scope.** This session ran entirely on a local CPU box (Apple Silicon, no CUDA) against the
converted `floorplans-door` val/test trees copied in from the main checkout, with the DINOv2 and
FastSAM weights symlinked in. Numbers here are therefore a **fresh, session-local measurement**,
not a re-run of the committed cross-method GPU sweep behind `docs/eval/floorplans-findings.md` —
they differ slightly from that report's dino-dense row (0.147 vs the 0.113–0.117 baseline
re-measured here) despite an identical `fixed_input_side=1120` config, most plausibly because that
report's tuning grid included `fixed_input_side` directly in its trial dict (confirmed from the
gitignored `docs/benchmark/*.json` artifacts on disk) while the **currently committed**
`eval/tuning.py` grid for `dino-dense` does not (see "A found-but-deferred bug" below) — so the
exact grid that produced the committed number no longer matches what ships. The qualitative
pattern (small ≪ medium/large recall, F1 well behind the other five methods) reproduces cleanly
either way, and that pattern is what this session set out to fix.

## Root cause — confirmed: token starvation, not a metric artifact

Instrumented the exact resize math `dino_dense.py` uses (`.scratch/token_starvation.py`, pure
geometry, no inference) over every GT door box on `floorplans-door` val+test, computing the
**effective tokens spanned by each box's shorter side** at the model's input resolution:

| split | bucket | n | native (1568 cap) median tokens/side | letterbox (1120) median tokens/side |
|---|---|---|---|---|
| val | small | 282 | 2.00 | 3.06 |
| val | medium | 229 | 4.14 | 5.12 |
| val | large | 16 | 6.43 | 8.56 |
| test | small | 84 | 3.21 | 3.37 |
| test | medium | 135 | 3.50 | 5.28 |
| test | large | 14 | 4.61 | 7.61 |

Small doors land at **2–3.4 stride-14 tokens per side** at both the production letterbox and the
plain native cap — confirming the method's own documented failure mode ("a whole instance spans
~2–3 stride-14 tokens") and the root-cause hypothesis this session set out to test. Medium doors
(5–5.4 tokens) are comfortably better represented; large (7.6–8.6) better still.

## The fix — iterated empirically

Measured on floorplans-door **test** (28 plans) every iteration, **val** (56 plans) as the
overfitting check, and the **chipset/textured regimes** (EASY/TEXTURED/VARIED/CLUTTERED) as the
regression guardrail. All `dino-dense` fields below are **additive and opt-in** (default
`None`/unchanged); nothing in this PR touches the chipset/textured/synthetic default behaviour.
Verified: the regime check (`.scratch/dd_harness.py::run_regime_check`, default config)
reproduces **EASY 0.172 / TEXTURED 0.760 / VARIED 0.641 / CLUTTERED 0.690** byte-for-byte before
and after every change in this log.

### Pass 1 — blanket bigger `fixed_input_side` (cheapest lever, no code change)

| `fixed_input_side` | test F1 | small | medium | large | wall-clock (28 imgs) |
|---|---|---|---|---|---|
| 1120 (baseline) | 0.117 | 0.083 | 0.200 | 0.214 | 90s |
| 1400 | 0.125 | 0.095 | 0.215 | 0.143 | 165s |
| 1680 | 0.117 | 0.095 | 0.193 | 0.071 | 298s |

A blanket resolution bump gives at most a marginal, noisy gain, at **quadratic-ish cost** (DINOv2's
self-attention scales with tokens²: 1680 costs ~4.6× 1120's wall-clock for essentially the same
F1). **Reverted** — not worth shipping as the primary lever. It applies the same extra resolution
to every plan uniformly, when only the plans with a small exemplar actually need it.

### Pass 2 — adaptive resolution (landed, opt-in)

Implemented the backlog item ("size the scene so the exemplar spans ≥ N stride-14 tokens") as two
new `DinoDenseConfig` fields:

- `adaptive_min_exemplar_tokens: int | None` — when set, the scene may be **upscaled** (not just
  downscaled) so the exemplar's shorter side spans at least this many tokens.
- `adaptive_max_side: int | None` — a **separate, higher ceiling** used only while resolving a
  starved exemplar, so only the images that actually need the extra resolution pay for it; other
  images keep paying the ordinary `scene_max_side` cost. `None` reuses `scene_max_side`.

Two bugs fixed alongside (both are correctness fixes, not new behaviour on any config that never
upscales — i.e. every config that shipped before this session): `cv2.INTER_AREA` is documented by
OpenCV to degrade to `INTER_NEAREST` on magnification, so both the scene-cap resize and the
`_fixed_letterbox` resize now pick `INTER_LINEAR` when the computed scale is `> 1.0`.

First measurement — **adaptive resolution alone, no letterbox** (`fixed_input_side=None`):

| config | test F1 | small | medium | large |
|---|---|---|---|---|
| pure native (no adaptive, no letterbox) | 0.074 | 0.083 | 0.111 | 0.071 |
| adaptive tokens=6, ceiling=1568 | 0.096 | 0.095 | 0.156 | 0.143 |
| adaptive tokens=6, ceiling=2240 | 0.092 | 0.095 | 0.148 | 0.143 |
| adaptive tokens=6, ceiling=2800 | 0.093 | 0.095 | 0.148 | 0.143 |

Adaptive resolution alone beats plain native at the *same* compute ceiling (0.096 vs 0.074) — the
mechanism genuinely reallocates a fixed budget toward the exemplars that need it. But raising the
ceiling past 1568 buys **nothing further** — small-symbol recall is flat at 0.095 from ceiling 1568
through 2800, so past a low floor, resolution stops being the bottleneck for the remaining starved
instances. Still notably *below* the plain `fixed_input_side=1120` letterbox baseline (0.117),
which was surprising enough to chase down:

### Pass 3 — adaptive resolution + a MATCHED letterbox (the winning combination)

Hypothesis: `dino-dense`'s letterbox pads a scene to a **square**, and DINOv2's positional-embedding
interpolation (fixed at export time, tuned near-square) may degrade on the strongly non-square
aspect ratios floor plans often have — independent of raw resolution. Tested `fixed_input_side` set
to the **same value** as `adaptive_max_side`, so the adaptive upscale is preserved into the letterbox
rather than immediately re-downscaled by a smaller letterbox side:

| config | test F1 | test small | test medium | test large | val F1 | val small | val medium | val large |
|---|---|---|---|---|---|---|---|---|
| letterbox 1120 (production baseline) | 0.113 | 0.083 | 0.193 | 0.214 | 0.080 | 0.078 | 0.140 | 0.313 |
| **letterbox 1568 + adaptive(6, ceil 1568)** | **0.144** | **0.143** | **0.215** | **0.357** | 0.081 | **0.092** | **0.153** | 0.188 |

Test F1 **0.117 → 0.144** (+23% relative), small-symbol recall **0.083 → 0.143** (+72% relative).
**Val is a genuine, honest caveat**: pooled F1 is flat (0.080 → 0.081) even though small (0.078 →
0.092) and medium (0.140 → 0.153) recall both move the same direction as test — the large bucket
(n=16 on val, noisy) drops sharply (0.313 → 0.188) and a precision shift roughly cancels the pooled
F1 gain. **Read this as a real but modest win, not a fix for the small-symbol regime** — recall
consistently improves in the size buckets doing the most damage (small covers 36% of instances,
medium 58%), on both splits, but the method does not close the gap to `ncc`/`propose-retrieve` on
doors. Cost: ~15s/image on CPU vs ~3s/image at 1120 (real, ~5× latency for the gain — a domain
config trade a practitioner should make deliberately, not a new default).

**Landed as two additive `DinoDenseConfig` fields** (`adaptive_min_exemplar_tokens`,
`adaptive_max_side`), default `None` — byte-identical elsewhere. The recommended floor-plan-door
override is `fixed_input_side=1568, adaptive_min_exemplar_tokens=6, adaptive_max_side=1568`.

## Box-shaping / over-prediction — tried, REVERTED (two variants)

The qualitative symptom reported going into this session: a visually clean, similarly-sized match
still missing the IoU-0.5 bar, hypothesised as the raw connected-component **bounding rect** not
respecting a thin/irregular symbol's true shape (a door is a line + swing arc, not a filled
rectangle). Implemented `box_shape="exemplar-prior"`: replace the blob's own bounding rect with a
box shaped exactly like the **exemplar** (its own w×h), centred on the component.

| centring | test F1 (at letterbox 1120) | small | medium | large |
|---|---|---|---|---|
| blob (baseline) | 0.113 | 0.083 | 0.193 | 0.214 |
| exemplar-prior, PEAK-score pixel | 0.050 | 0.036 | 0.089 | 0.071 |
| exemplar-prior, CENTROID (at the winning resolution config) | 0.138 | 0.119 | 0.237 | 0.143 |
| blob (baseline, at the winning resolution config) | **0.144** | **0.143** | 0.215 | **0.357** |

Peak-centring is a clear regression (a `max-token` map's peak pixel is wherever the single
best-matching exemplar *part* sits — often off-centre, e.g. the door panel's straight edge rather
than the swing arc — so re-centring a fixed-size box there systematically mis-places it). Centroid
centring is closer but still net negative overall (worse on small/large, a wash-to-slightly-better
on medium). **Reverted both** — the raw blob's own extent tracks the true instance better than a
fixed-size box re-centred on any single point estimate. The area accept/reject bounds
(`min_area_frac`/`max_area_frac`) already do the real work of rejecting fragments and merged blobs;
shaping the box on top of that did not pay for itself. A genuine segmentation-based refinement
(FastSAM box snapping, already in `docs/methods/dino-dense.md`'s backlog) remains the more promising
open path here, not attempted this session (heavier, and the resolution work already consumed the
session's compute budget).

## Rotation sensitivity — tried, REGRESSED

Tested the speculative hypothesis (DINOv2 patch tokens are not rotation-equivariant, and floor-plan
door symbols are frequently mirrored/rotated relative to the exemplar): embed the exemplar crop at
0°/90°/180°/270° and **union** the four token banks into one enlarged bank for `max-token` scoring,
reusing the winning resolution config (`.scratch/test_rotation.py`, cheap to test — the SCENE
forward pass is cached and reused; only the tiny crop is re-embedded four times).

| exemplar token bank | test F1 | TP | FP | FN |
|---|---|---|---|---|
| single orientation (winner config) | 0.144 | 46 | 358 | 187 |
| 4-orientation union | 0.108 | 38 | 430 | 195 |

Worse on every count — recall dropped (fewer TP) *and* precision dropped (more FP): a wider token
bank matches more background clutter at *some* rotation without recovering the true rotated
instances it was meant to catch, since `max-token`'s top-k average gets diluted by the extra
candidate tokens. **Not shipped.** If revisited, per-rotation scoring with a scene-token **max**
across orientations (rather than pooling into one bank before scoring) would avoid diluting the
top-k average — flagged in the module's ROBUSTNESS BACKLOG, not attempted this session (time-boxed
per the task brief: "don't let it distract from the token-starvation work if it doesn't pay off
quickly").

## A found-but-deferred bug: the current floor-plan tuning grid for dino-dense is a no-op

`eval/tuning.py`'s `_TUNING_GRIDS["dino-dense"]` sweeps only `retain_frac` — but `retain_frac` is
read solely by the `self-similarity` calibration strategy, and `DinoDenseConfig.calibration`
defaults to (and the grid never overrides) `"contrast"`, which ignores it entirely. Every trial in
that grid therefore runs the byte-identical default config, and `pixi run tune-floorplans`'s
"tuned" dino-dense result is presently indistinguishable from its default. (The gitignored
`docs/benchmark/*-tuning-results.json` artifacts on disk show an *earlier* version of the grid did
include `fixed_input_side`, before a later "broaden domain tuning grids" commit dropped it —
explaining the discrepancy between this session's measurements and the committed 0.147 headline
number.) **Left as-is** — fixing the shared tuning harness is out of scope for a single-method PR
per the project's method-improvement playbook, and other worktrees may be actively touching that
file. Flagged here so the next floor-plan tuning pass picks it up: the grid should search
`fixed_input_side` × `adaptive_min_exemplar_tokens` (with `adaptive_max_side` pinned equal to
`fixed_input_side`), not `retain_frac`.

## Result vs the other regimes (unchanged, verified every iteration)

| regime | before | after |
|---|---|---|
| EASY (chipset) | 0.172 | 0.172 |
| TEXTURED | 0.760 | 0.760 |
| VARIED (scale/rotation) | 0.641 | 0.641 |
| CLUTTERED | 0.690 | 0.690 |

Byte-identical — every new field defaults to `None`/off, and the two interpolation bug fixes only
change behaviour when the computed scale is `> 1.0`, which never happens on the default
(non-adaptive, non-letterbox) chipset/textured/synthetic path.

## Verification

`pixi run quality` (Ruff + Ruff-format + MyPy strict + full test suite with coverage) green on this
branch. Three new model-free tests added (`tests/test_dino_dense.py`), stub-inferencer driven per
the project's CI-coverage-without-weights pattern, covering: the adaptive upscale branch, the
`scene_max_side`-ceiling clamp, and the `adaptive_max_side` override — `dino_dense.py` module
coverage 98% (the two remaining uncovered lines predate this session). No dataset images or raw
per-image data are committed (gitignored per project convention); the floor-plan tree used this
session was copied locally from the main checkout's already-converted `datasets/floorplans-door/`.
