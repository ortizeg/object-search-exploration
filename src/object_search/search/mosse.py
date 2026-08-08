"""Method -- MOSSE/ASEF correlation-filter matching via FFT (the fast cousin of ``ncc``).

What it does
------------
``ncc`` finds rotated repeats with a **brute-force bank**: it spatially correlates the raw crop
against the scene once per (scale, angle) pair, so its default 5-scale x 7-angle bank is 35
``cv2.matchTemplate`` passes and the 6000x4000 chipset costs ~68 s. This method attacks exactly
that cost. It **synthesizes a small bank of discriminative filters** from the same warped-exemplar
set (the rotation bank is folded into a few sharp filters via the closed-form MOSSE/ASEF solve),
then matches each with an **FFT cross-correlation per scale level** -- ~3 filters x 5 levels = 15
FFT correlations, not 35 spatial ones. FFT cross-correlation is ``O(H*W*log(H*W))`` versus spatial
matching's ``O(H*W*h*w)``, so the win compounds with template size and is decisive on the large
chipset. The filters are *regularized* correlation filters (MOSSE / ASEF, Bolme et al. 2010), not
raw matched filters: the closed-form solve whitens the exemplar and learns to suppress its own
sidelobes, which is what lets one filter span a trained rotation sub-range without a per-angle pass.

Why a *small bank* and not a single filter: folding the whole +/-35 deg rotation range into ONE
filter blurs it (the average of seven orientations matches none of them crisply), which is measured
to lose the sharp peaks that separate near-identical repeats from their inter-instance sidelobes.
Folding it into a few (default 3) *sharp* sub-filters -- each covering a contiguous angle sub-range
-- and taking the per-pixel max of their responses recovers the sharp peaks AND spans the rotation
range, at a fraction of ncc's spatial cost. That is the measured sweet spot between one blurry
filter and ncc's seven separate spatial passes (see ``docs/reports/mosse-improvement.md``).

This file is meant to be read top to bottom by an ML practitioner. Readability outranks DRY
(project convention): the numbered steps ``# 1.`` .. ``# 9.`` in :func:`search` match the
headings in ``docs/methods/mosse.md`` one-for-one, and a step may inline a few lines rather than
reach for a shared helper if that reads better standalone.

The filter bank, step by step
-----------------------------
At each scale level, for each contiguous angle sub-range (a "group"):

1. Warp the crop across that group's angles x ``train_scales`` -- this sub-range is the filter's
   "training set", drawn from the same warp bank ``ncc`` correlates one member at a time.
2. For each warped patch ``f_i`` (preprocessed, below) take ``F_i = FFT2(f_i)``; the desired
   output is a Gaussian ``g`` (peaked at the array origin) with ``G = FFT2(g)``.
3. Pool the MOSSE numerator and denominator across the sub-bank and solve in closed form::

       H = (sum_i G . conj(F_i)) / (sum_i F_i . conj(F_i) + eps)

   (``.`` is elementwise). Pooling numerator and denominator *separately* is what makes this
   MOSSE rather than the noise-fragile ASEF average-of-exact-filters or the overfit MACE. The
   spatial filter is ``k = real(IFFT2(conj(H)))``, then DC-removed and unit-normalized.

Detection is then a normalized FFT cross-correlation of each ``k`` over the whole scene, combined
by per-pixel max across the bank (step 4).

Pre-processing (exact)
----------------------
- Colour: the BGR scene is converted **once** to single-channel grayscale with
  ``cv2.COLOR_BGR2GRAY`` (never per-channel colour correlation).
- Intensity, filter side: each training patch is ``log1p``-transformed (``log_transform``,
  default on -- the MOSSE illumination step, compressing multiplicative lighting into an
  additive offset the zero-mean filter then rejects), then normalized to zero-mean/unit-norm,
  then multiplied by a 2-D **Hann window** (``window``, default on) so the patch tapers to zero
  at the border. The window is mandatory for a clean filter because FFT correlation is
  **circular**: without it the patch's opposite edges wrap into a discontinuity that pollutes
  every frequency (Bolme et al. 2010, sec. 3).
- Intensity, scene side: the scene is ``log1p``-transformed to the **same space** as the filter
  (so the normalized response is a cosine in log-space) and kept ``float32``. It is **not**
  windowed -- windowing is a per-training-patch operation, not a scene operation.
- Search extent: the **full** scene, always (a cropped search would change the FFT and the
  local-energy normalization, breaking "same input => identical results").

Post-processing (exact)
-----------------------
- **Normalized response.** A raw correlation filter loses NCC's per-window normalization, so its
  response scales with local scene energy and there is no ``~1.0`` self-match anchor. This method
  restores the anchor cheaply: because ``k`` is DC-free, the numerator ``sum k . window`` equals
  ``sum k . (window - mean(window))``, so dividing by the window's L2 energy
  ``sqrt(sum(window^2) - sum(window)^2 / N)`` -- computed in ``O(H*W)`` with box filters --
  yields a cosine-like response in ``[-1, 1]``. That normalized response is what the PROPOSAL
  stage (peak extraction) works on, exactly where ``ncc`` uses ``TM_CCOEFF_NORMED``.
- **Coarse-to-fine verify (default on, step 6b).** The whitened filter is a strong LOCALIZER but a
  weak DISCRIMINATOR in clutter -- it proposes the true instances (measured: it peaks on ~83% of
  cluttered instances) but scores them alongside clutter, so a threshold on the filter response
  alone drops them. Each proposed peak is therefore re-scored by a LOCAL raw ``TM_CCOEFF_NORMED``
  of the rotated exemplar in a small window around the box -- ``ncc``'s discriminative score and
  its clean ``~1.0`` self-anchor, but evaluated only at the handful of proposal sites, never over
  the whole scene. That raw score drives the threshold and the candidate log. This is the "fine"
  half of a coarse-to-fine detector (the FFT filter is the cheap full-scene PROPOSER); it lifts
  CLUTTERED F1 from ~0.61 to ~0.82 and VARIED past ``ncc``, at the cost of a small EASY precision
  dip (raw NCC has periodic sidelobes on an identical-chip grid that the whitened filter
  suppressed). ``verify=False`` recovers the pure-filter response as a control. With
  ``mirror=True`` (**off** by default) the verify bank also correlates each angle's horizontally
  flipped exemplar (``cv2.flip``, template and mask together) -- scoped to this verify step only,
  since a reflection is not in the rotation group and no ``train_angles_deg`` bank, however wide,
  can ever reach it (the archetype: a floor-plan door drawn with the opposite swing hand).
- **PSR.** The peak-to-sidelobe ratio (MOSSE's native confidence: ``(peak - mu_side) /
  sigma_side`` over an annulus around the peak) is carried in diagnostics for inspection; the
  accept decision is driven by the normalized response, which is comparable across images.
- Response index ``(row, col)`` maps to ``BBox(x=col, y=row, w=tw, h=th)`` at that level's scale
  (top-left anchored, no centre offset); at scale ``s`` the original-image box divides back by
  ``s`` -- identical geometry to ``ncc`` (PITFALLS.md 1.2).
- Cross-level comparison is standardized: each level's map is z-scored against its own median/MAD
  before peaks are compared or suppressed across levels (a monotone transform, so peak locations
  are unchanged); the raw normalized response is carried alongside for the threshold and log.
- Threshold: the ``repeat-aware`` rule (default), re-anchored on the filter's self-response
  (which is **not** ``1.0`` -- the filter is a whitened exemplar, not the exemplar). It reads the
  distribution shape, never the ground-truth boxes; the same rule runs on every dataset. The
  shared ``self-similarity``/``ratio``/``gmm`` strategies stay selectable.
- Suppression + candidate log: identical to ``ncc`` -- cross-level greedy IoU NMS over the
  accepted matches, and a deduplicated sub-threshold candidate log (EVAL-08) so
  ``matches + candidates`` is one clean ranked detection set.

Known failure modes
--------------------
- **Textureless crop.** A flat exemplar makes the filter degenerate (the denominator is dominated
  by ``eps`` and the response is meaningless). Step 1 guards on the crop's std and abstains with
  ``outcome=EMPTY``, exactly as ``ncc`` does.
- **Scale beyond the pyramid.** Correlation is shift-invariant but **not** scale-invariant, and a
  single filter cannot span a wide scale range, so a scale pyramid is still required (the rotation
  bank is folded into the filter, but the scale bank is not). Instances scaled past ``scales`` are
  missed -- the log-polar / Fourier-Mellin front end in the ROBUSTNESS BACKLOG is the one-shot
  scale+rotation alternative.
- **Sharpness vs generalization.** A sharper filter (smaller ``output_sigma`` / ``regularization``)
  localizes crisply but misses off-training angles; a broader one generalizes but drops precision
  in clutter. This is the genuine knob OTSDF exists for; the defaults sit where the synthetic
  splits measured best.
- **Lighting/pose change.** ``log1p`` + the DC-free filter give real illumination robustness that
  ``ncc`` lacks, but a large out-of-plane pose change still drops the correlation.

ROBUSTNESS BACKLOG
------------------
Deferred deliberately (mirrored verbatim in ``docs/methods/mosse.md`` and
``docs/ROBUSTNESS-BACKLOG.md``):

- **Log-polar / Fourier-Mellin front end** so one correlation spans rotation **and** scale,
  retiring the scale pyramid entirely (the rotation bank is already folded into the filter).
- **A dedicated DSST-style scale filter** -- a separate 1-D correlation filter over a scale
  pyramid of the peak patch -- for continuous scale estimation instead of the discrete pyramid.
- **OTSDF / UMACE variants** exposing an explicit sharpness-vs-noise trade-off parameter, for
  scenes where the MOSSE default is either too sharp (misses poses) or too broad (clutter FPs).
- **Kernelized correlation filters (KCF)** -- a non-linear kernel in the closed-form solve, more
  discriminative against structured background than the linear MOSSE filter here.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Literal

import cv2
import numpy as np
import numpy.typing as npt
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field
from scipy import fft

from object_search.schemas import (
    BBox,
    Candidate,
    Diagnostics,
    ExemplarBox,
    HeatmapPayload,
    LatencyBreakdown,
    Match,
    SearchOutcome,
    SearchResult,
)
from object_search.search.common import calibration, nms, peaks, viz
from object_search.search.registry import register_method

# -- Tunables that are properties of the METHOD, not of a query, so they are module -------
# -- constants rather than config fields (a config field the user cannot sensibly move is --
# -- noise on the UI form). Each is justified from the measured research. -----------------

_METHOD_VERSION = "1.0.0"
# Below this the crop is textureless: the filter's denominator collapses to the regularizer and
# the response is noise. A uniform crop has std exactly 0.0; the epsilon catches float dust.
_TEMPLATE_STD_FLOOR = 1e-6
# A sub-8px filter is statistically worthless and the FFT solve is dominated by the border taper
# (PITFALLS.md 1.4/1.5). Levels producing one are skipped.
_MIN_TEMPLATE_PX = 8
# Peaks are picked on the per-level z-score, so the floor is in standard deviations above the
# level's own noise -- 3 sigma, matching ncc so cross-level comparison is honest.
_PEAK_Z_FLOOR = 3.0
_MAD_TO_STD = 1.4826  # median-absolute-deviation -> std, for a Gaussian
# A peak whose box overlaps the exemplar by at least this is the exemplar's own self-match,
# labelled is_exemplar rather than dropped or double-counted (METHOD-04c).
_EXEMPLAR_IOU = 0.5
# repeat-aware calibration fractions. These are tuned for the DEFAULT verify=True score (a raw
# local NCC, self-match ~1.0 -- ncc's anchor, restored by step 6b), NOT the bare filter response.
# near_frac 0.9 is deliberately high: a cluttered instance's verify score sits well below the
# self-match, so counting only records >= 0.9 x self as "near-identical repeats" keeps clutter on
# the PERMISSIVE path (self x retain_frac); at 0.85 the strict cut fired on cluttered scenes and
# crushed recall (measured CLUTTERED R 0.60 -> 0.76 going 0.85 -> 0.9). With verify=False the score
# reverts to the lower, image-dependent bare-filter response and these fractions are only
# approximate -- verify=False is an ablation control, not the tuned path. See
# _repeat_aware_threshold and docs/reports/mosse-improvement.md (iteration 2).
_REPEAT_NEAR_FRAC = 0.9
_REPEAT_STRICT_FRAC = 0.8
# PSR sidelobe geometry: exclude an (2r+1)^2 window around the peak, measure the mean/std of the
# rest. 11px is Bolme et al.'s default exclusion radius; purely a diagnostic here.
_PSR_EXCLUDE_RADIUS = 11
# Coarse-to-fine verify (step 6b): grow each proposal box by this fraction of the template size on
# every side before the local NCC re-score, so the raw correlation absorbs the sub-pixel drift the
# pyramid rounding leaves (the response argmax divides back by the level scale). This MUST stay
# small: a large margin lets the window reach a *neighbouring* instance in a packed grid, whose ~1.0
# correlation would then inflate a wrong proposal's score into a false positive (measured: a 0.5
# margin collapsed TEXTURED precision to 0.34). 0.15 (a few px) covers the rounding, no more.
_VERIFY_MARGIN_FRAC = 0.15


class MOSSEConfig(BaseModel):
    """Frozen config for :func:`search`; its JSON Schema drives the UI form (one source).

    Every field carries a ``description`` because that text becomes the form's help string --
    this is the single place it is written, so it must be written here.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    scales: tuple[float, ...] = Field(
        default=(0.75, 0.875, 1.0, 1.15, 1.3),
        description=(
            "Pyramid scale factors. Correlation is shift- but NOT scale-invariant and one filter "
            "cannot span a wide scale range, so a pyramid is still required. The SCENE is resized "
            "by each factor and the filter is built from the exemplar cropped from that resized "
            "scene, which keeps detection geometry identical to ncc. One FFT correlation per level."
        ),
    )
    train_angles_deg: tuple[float, ...] = Field(
        default=(-35.0, -23.3, -11.7, 0.0, 11.7, 23.3, 35.0),
        description=(
            "Rotation bank folded INTO the single filter (ncc's bank, but paid once). The crop is "
            "warped to each angle and all warps are averaged into one MOSSE filter, so rotated "
            "repeats are found with one FFT correlation instead of one spatial pass per angle. "
            "Default is a 7-step bank over +/-35 deg (~11.7 deg spacing), matching ncc's default."
        ),
    )
    train_scales: tuple[float, ...] = Field(
        default=(1.0,),
        description=(
            "Optional scale jitter folded into the filter alongside the rotation bank, widening "
            "its scale tolerance so the pyramid levels can be spaced further apart. Default (1.0,) "
            "= no jitter (the pyramid handles scale); (0.9, 1.0, 1.1) broadens the filter."
        ),
    )
    n_angle_groups: int = Field(
        default=3,
        ge=1,
        description=(
            "How many sharp sub-filters the rotation bank is split into (a small filter BANK). "
            "Each sub-filter is built from a contiguous angle sub-range and correlated separately; "
            "the per-pixel MAX of their responses combines them. 1 = one filter averaged over all "
            "angles (fast but blurry -- loses TEXTURED sharpness); more groups = sharper "
            "sub-filters (crisper peaks, better rotated-instance recall) but more FFT correlations "
            "and, past ~3, over-sharp filters that miss tiny objects. 3 measured best -- the sweet "
            "spot between one blurry filter and ncc's 7 separate spatial passes."
        ),
    )
    output_sigma: float = Field(
        default=1.0,
        gt=0.0,
        description=(
            "Std (pixels) of the Gaussian correlation target g the filter is solved to produce. "
            "Smaller = a sharper peak (crisper localization, but misses off-training angles); "
            "larger = a broader, more forgiving filter. The MOSSE sharpness knob."
        ),
    )
    regularization: float = Field(
        default=0.3,
        gt=0.0,
        description=(
            "MOSSE denominator regularizer eps, RELATIVE to the mean filter-energy, that keeps the "
            "closed-form solve stable at frequencies the exemplar has no energy in. Larger = a "
            "broader, more noise-robust but less sharp filter (the numerically-stable descendant "
            "of MACE, which sets eps->0 and overfits)."
        ),
    )
    energy_floor_frac: float = Field(
        default=0.7,
        ge=0.0,
        description=(
            "Floor added to the local-energy denominator, as a fraction of the scene's MEDIAN "
            "window energy, so a flat low-energy region cannot divide a near-zero numerator up "
            "into a spurious ~1.0 response (the correlation-filter analogue of the degenerate "
            "TM_CCOEFF_NORMED flat-window case). 0.7 measured best (a broad 0.5-0.8 plateau); the "
            "original 0.3 left spurious background peaks on the tiny-chip regime, costing EASY "
            "precision -- raising the floor takes EASY precision to 1.00 and TEXTURED to a perfect "
            "F1 (see docs/reports/mosse-improvement.md, iteration 2). 0.0 disables the floor."
        ),
    )
    log_transform: bool = Field(
        default=True,
        description=(
            "Apply log1p to filter patches AND the scene before correlation (MOSSE illumination "
            "step): compresses multiplicative lighting into an additive offset the DC-free filter "
            "rejects. This is the local-normalization ncc gets from TM_CCOEFF_NORMED; off = raw "
            "intensities (a control for measuring the log step's contribution)."
        ),
    )
    window: bool = Field(
        default=True,
        description=(
            "Multiply each training patch by a 2-D Hann window before the FFT. FFT correlation is "
            "circular, so without the taper the patch's opposite edges wrap into a discontinuity "
            "that pollutes the filter. On by default; off is a control to show the artifact."
        ),
    )
    verify: bool = Field(
        default=True,
        description=(
            "Coarse-to-fine re-scoring. The whitened MOSSE filter localizes well but is a weak "
            "DISCRIMINATOR in clutter -- it proposes true instances (measured: it peaks on ~83% of "
            "cluttered instances) but scores them alongside clutter so the threshold drops them. "
            "With verify on, every proposed peak is re-scored by a LOCAL raw TM_CCOEFF_NORMED of "
            "the rotated exemplar (ncc's score, but evaluated only in a small window "
            "around each of the few proposals -- O(peaks) local passes, NOT ncc's full-scene "
            "scale x angle sweep), and that raw score drives the threshold + candidate log. It is "
            "the 'fine' half of a coarse-to-fine detector: the FFT filter is the cheap full-scene "
            "PROPOSER, the local NCC is the accurate VERIFIER. It restores ncc's ~1.0 self-anchor "
            "and its clutter discrimination while keeping the filter's speed. off = the pure "
            "filter response (a control, and the original shipped behaviour)."
        ),
    )
    mirror: bool = Field(
        default=False,
        description=(
            "Also verify against the horizontally MIRRORED exemplar at every angle in "
            "train_angles_deg. Off by default. Scoped to the coarse-to-fine VERIFY step only (a "
            "local raw NCC re-score, ncc's own mirror mechanism) -- it does NOT fold mirrored "
            "angles into the FFT filter's training warp, since the filter proposes candidates by "
            "shape regardless of handedness and the verify step is what actually discriminates "
            "them by score. A reflection is not in the rotation group, so no train_angles_deg bank "
            "-- however wide -- can ever reach it; the archetype is a floor-plan door drawn with "
            "the opposite swing hand. Inert (a no-op) when verify=False, since there is then no "
            "local re-score for it to extend."
        ),
    )
    threshold: float | None = Field(
        default=None,
        description="Fixed accept threshold on the normalized response. None => calibrate it.",
    )
    calibration: Literal["fixed", "self-similarity", "ratio", "gmm", "repeat-aware"] = Field(
        default="repeat-aware",
        description=(
            "How the accept threshold is chosen when `threshold` is None. repeat-aware (default) "
            "reads the score distribution: >=2 distinct locations near the filter's self-response "
            "=> near-identical repeats => strict cut; else transformed instances => permissive "
            "self_score * retain_frac tail. Re-anchored for the filter (self-response is not 1.0). "
            "self-similarity is the plain fixed-fraction cut; ratio/gmm are the controls."
        ),
    )
    peaks: Literal["nms", "local-max", "watershed"] = Field(
        default="local-max",
        description=(
            "Peak-extraction strategy. local-max (default) separates touching instances that "
            "plain nms merges; nms is the control; watershed uses a distance transform."
        ),
    )
    nms_iou: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="IoU above which two accepted boxes are suppressed to one (cross-level NMS).",
    )
    suppression_radius_frac: float = Field(
        default=0.5,
        gt=0.0,
        description="local-max footprint as a fraction of the template size (size-aware).",
    )
    max_candidates: int = Field(
        default=50,
        ge=1,
        description="How many top peaks (with raw scores) to keep for the EVAL-08 candidate log.",
    )
    seed: int = Field(
        default=0,
        ge=0,
        description="random_state for the gmm calibrator (its only genuinely stochastic step).",
    )
    retain_frac: float = Field(
        default=0.35,
        gt=0.0,
        le=1.0,
        description=(
            "The permissive self-relative accept fraction: keep a match above "
            "self_score * retain_frac. Used directly by self-similarity and as the "
            "transformed-instance floor by repeat-aware. Tuned on the synthetic splits to the "
            "shape of the score distribution, NOT to the ground-truth boxes. 0.35 fits the "
            "DEFAULT verify=True score (a raw local NCC with a ~1.0 self-anchor): a cluttered/"
            "transformed instance's raw correlation sits around 0.35-0.5 x self, so 0.35 admits it "
            "while rejecting clutter below (measured CLUTTERED F1 0.61 -> 0.82). The old 0.5 fit "
            "the un-verified filter response; too strict for the verified score, it drops recall."
        ),
    )


@dataclass(frozen=True)
class _LevelPeak:
    """One extracted peak carried through post-processing with everything the split needs.

    ``raw_score`` is the normalized correlation-filter response in ``[-1, 1]`` (what the threshold
    and the candidate log use); ``z_score`` is the per-level-standardised value (what cross-level
    ranking and NMS use, since raw scores are not comparable across template sizes).
    """

    box: BBox
    raw_score: float
    z_score: float
    level: float


def _gaussian_target(h: int, w: int, sigma: float) -> npt.NDArray[np.float64]:
    """The desired correlation output ``g``: a unit-peak Gaussian at the patch centre.

    Built centred and then ``ifftshift``-ed so its peak sits at the array origin ``(0, 0)``. That
    origin placement makes ``G = FFT2(g)`` zero-phase, so the spatial filter recovered from the
    closed-form solve (``k = real(IFFT2(conj(H)))``, step 3) is a centred, appearance-aligned
    template, which is what makes the whole-scene ``valid`` cross-correlation top-left anchored
    like ncc -- verified against the exemplar self-match localizing to its own box.
    """
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma * sigma))
    return np.asarray(np.fft.ifftshift(g), dtype=np.float64)


def _preprocess_patch(
    patch: npt.NDArray[np.float64], *, log_transform: bool, hann: npt.NDArray[np.float64] | None
) -> npt.NDArray[np.float64]:
    """MOSSE patch preprocessing: log1p -> zero-mean/unit-norm -> Hann window.

    ``log1p`` compresses multiplicative illumination; zero-mean/unit-norm makes the filter energy
    comparable across the bank; the Hann window tapers the border to zero so the circular FFT does
    not wrap a discontinuity into every frequency (Bolme et al. 2010, sec. 3).
    """
    proc = np.log1p(patch) if log_transform else patch.astype(np.float64)
    proc = proc - float(proc.mean())
    norm = float(np.linalg.norm(proc))
    if norm > 0.0:
        proc = proc / norm
    if hann is not None:
        proc = proc * hann
    return proc


def _angle_groups(angles_deg: tuple[float, ...], n_groups: int) -> list[tuple[float, ...]]:
    """Split the rotation bank into ``n_groups`` contiguous, roughly-even angle sub-ranges.

    Sorting first means each group covers an adjacent slice of the rotation range (e.g. 7 angles
    into 3 groups -> [-35, -23.3] / [-11.7, 0] / [11.7, 23.3, 35]), so every sub-filter is built
    from *near* orientations and stays sharp, while the group set still spans the whole range. A
    filter built from angles scattered across the range would be as blurry as one averaged filter,
    which defeats the point of the bank.
    """
    srt = sorted(angles_deg)
    g = max(1, min(n_groups, len(srt)))
    return [tuple(srt[i * len(srt) // g : (i + 1) * len(srt) // g]) for i in range(g)]


def _build_filter(
    base_template: npt.NDArray[np.uint8],
    *,
    angles_deg: tuple[float, ...],
    train_scales: tuple[float, ...],
    output_sigma: float,
    regularization: float,
    log_transform: bool,
    window: bool,
) -> npt.NDArray[np.float32]:
    """Solve one MOSSE/ASEF filter from a warped-exemplar sub-bank (the method's heart).

    The rotation x scale warp bank is the filter's training set. Each warped patch ``f_i`` and the
    common Gaussian target ``g`` contribute to the pooled MOSSE numerator/denominator; the closed
    form ``H = sum(G . conj(F_i)) / (sum(F_i . conj(F_i)) + eps)`` is solved once. The spatial
    filter ``k = real(IFFT2(conj(H)))`` is then DC-removed (so a scene brightness offset is
    rejected and the normalized-response numerator is a clean ``sum k . (window - mean)``) and
    unit-normalized (so the response denominator is just the window energy). ``angles_deg`` is one
    contiguous sub-range from :func:`_angle_groups` -- the whole bank folds several of these.
    """
    h, w = base_template.shape[:2]
    template = base_template.astype(np.float64)
    hann = np.outer(np.hanning(h), np.hanning(w)) if window else None

    g = _gaussian_target(h, w, output_sigma)
    g_fft = np.fft.fft2(g)

    num = np.zeros((h, w), dtype=np.complex128)
    den = np.zeros((h, w), dtype=np.complex128)
    for angle in angles_deg:
        for scale in train_scales:
            # Warp the crop about its centre, staying in the SAME (h, w) frame so every f_i shares
            # the filter's shape. INTER_LINEAR + reflect border avoids the fabricated-corner zeros
            # that would otherwise bias the filter toward flat regions.
            rot = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
            warped = np.asarray(
                cv2.warpAffine(
                    template, rot, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
                ),
                dtype=np.float64,
            )
            f = _preprocess_patch(warped, log_transform=log_transform, hann=hann)
            f_fft = np.fft.fft2(f)
            num += g_fft * np.conj(f_fft)
            den += f_fft * np.conj(f_fft)

    eps = regularization * float(den.real.mean())
    h_star = num / (den + eps)
    # The MOSSE detection response is ifft2(F_scene . H_star); the equivalent spatial correlation
    # template for a whole-scene `valid` cross-correlation is real(ifft2(conj(H_star))). With g
    # peaked at the array origin (above) this recovers a centred, appearance-aligned filter whose
    # correlation peak lands at the object's top-left -- verified against the exemplar self-match.
    kernel = np.real(np.fft.ifft2(np.conj(h_star)))
    kernel = kernel - float(kernel.mean())  # DC-free => brightness-invariant, clean numerator
    knorm = float(np.linalg.norm(kernel))
    if knorm > 0.0:
        kernel = kernel / knorm
    return np.ascontiguousarray(kernel, dtype=np.float32)


def _build_filter_bank(
    base_template: npt.NDArray[np.uint8], config: MOSSEConfig
) -> list[npt.NDArray[np.float32]]:
    """Build the small bank of sharp sub-filters, one per contiguous angle group (step 3).

    Splitting the rotation bank into a few sharp sub-filters (default 3) instead of one filter
    averaged over all angles is the measured sweet spot: one averaged filter is blurry and loses
    the crisp peaks that separate near-identical repeats from their inter-instance sidelobes, while
    ncc pays for one *separate spatial pass per angle*. A handful of FFT-correlated sub-filters
    keeps the peaks sharp AND spans the rotation range, at a fraction of ncc's cost (step 4).
    """
    groups = _angle_groups(config.train_angles_deg, config.n_angle_groups)
    return [
        _build_filter(
            base_template,
            angles_deg=group,
            train_scales=config.train_scales,
            output_sigma=config.output_sigma,
            regularization=config.regularization,
            log_transform=config.log_transform,
            window=config.window,
        )
        for group in groups
        if group
    ]


def _energy_map(
    scene64: npt.NDArray[np.float64], h: int, w: int, out_h: int, out_w: int, floor_frac: float
) -> npt.NDArray[np.float64]:
    """Per-window L2 energy of the mean-subtracted scene -- the normalization denominator.

    ``sqrt(sum(window^2) - sum(window)^2 / N)`` computed in ``O(H*W)`` with box filters (anchored
    at the window's top-left so output ``(r, c)`` is the energy of ``scene[r:r+h, c:c+w]`` -- the
    same top-left anchoring as the valid correlation). A floor of ``floor_frac`` x the median
    window energy is added so a flat low-energy region cannot divide a near-zero numerator up into
    a spurious ~1.0 peak (the correlation-filter analogue of the degenerate flat-window case). This
    depends only on the scene and the filter *size*, so it is computed ONCE per scale and shared
    across the whole sub-filter bank.
    """
    n = float(h * w)
    win_sum = cv2.boxFilter(
        scene64,
        ddepth=cv2.CV_64F,
        ksize=(w, h),
        anchor=(0, 0),
        normalize=False,
        borderType=cv2.BORDER_CONSTANT,
    )[:out_h, :out_w]
    win_sqsum = cv2.boxFilter(
        scene64 * scene64,
        ddepth=cv2.CV_64F,
        ksize=(w, h),
        anchor=(0, 0),
        normalize=False,
        borderType=cv2.BORDER_CONSTANT,
    )[:out_h, :out_w]
    energy = np.sqrt(np.maximum(win_sqsum - (win_sum * win_sum) / n, 0.0))
    return energy + floor_frac * float(np.median(energy)) + 1e-9


def _bank_response(
    scene: npt.NDArray[np.float32],
    kernels: list[npt.NDArray[np.float32]],
    energy_floor_frac: float,
) -> npt.NDArray[np.float64]:
    """Combined normalized response of the sub-filter bank -- per-pixel MAX over the sub-filters.

    Each sub-filter's numerator is a linear (zero-padded, NOT circular) cross-correlation of the
    DC-free filter over the scene, in ``O(H*W*log(H*W))`` -- the whole point of the method. The
    **scene's FFT is transformed once and reused** for every sub-filter (only the small kernel is
    re-transformed and the product re-inverted), and the local-energy denominator is computed once
    per scale; both matter because the forward FFT of a 24-megapixel chipset scene dominates the
    cost. Because each filter is DC-free and unit-norm, numerator/energy is a cosine in ``[-1, 1]``
    -- the correlation-filter analogue of ncc's ``TM_CCOEFF_NORMED``.

    Max (not sum) combines the bank because the sub-filters are alternatives: a given instance
    matches whichever sub-filter's angle sub-range covers its orientation, and taking the max keeps
    that sub-filter's sharp peak instead of diluting it with the others' near-zero responses there.
    """
    h, w = kernels[0].shape[:2]
    height, width = scene.shape[:2]
    out_h, out_w = height - h + 1, width - w + 1
    scene64 = scene.astype(np.float64)

    # Transform the scene ONCE at the linear-correlation (zero-padded) size, then correlate every
    # sub-filter by transforming only the small kernel and inverting the product -- the valid region
    # of the full correlation is [h-1:H, w-1:W], the same top-left anchoring as ncc.
    fshape = (fft.next_fast_len(height + h - 1), fft.next_fast_len(width + w - 1))
    scene_fft = fft.rfft2(scene64, s=fshape)
    energy = _energy_map(scene64, h, w, out_h, out_w, energy_floor_frac)

    def _one(kernel: npt.NDArray[np.float32]) -> npt.NDArray[np.float64]:
        # Cross-correlation = convolution with the flipped kernel; multiply in the shared spectrum.
        flipped = kernel[::-1, ::-1].astype(np.float64)
        full = fft.irfft2(scene_fft * fft.rfft2(flipped, s=fshape), s=fshape)
        clipped = np.clip(full[h - 1 : height, w - 1 : width] / energy, -1.0, 1.0)
        return np.asarray(clipped, dtype=np.float64)

    # kernels is never empty (n_angle_groups >= 1), so seed with the first and max the rest in.
    response = _one(kernels[0])
    for kernel in kernels[1:]:
        np.maximum(response, _one(kernel), out=response)
    return response


def _psr(response: npt.NDArray[np.float64], y: int, x: int) -> float:
    """Peak-to-sidelobe ratio at ``(y, x)`` -- MOSSE's native confidence (a diagnostic here).

    ``(peak - mean_sidelobe) / std_sidelobe`` over the response minus an exclusion window around
    the peak. High PSR (Bolme et al. suggest > ~7) means a sharp, unambiguous peak; low PSR means
    the filter is responding diffusely. Carried in diagnostics, not used for the accept decision
    (the normalized response transfers across images; a raw PSR does not as cleanly).
    """
    r = _PSR_EXCLUDE_RADIUS
    peak = float(response[y, x])
    mask = np.ones(response.shape, dtype=bool)
    y0, y1 = max(0, y - r), min(response.shape[0], y + r + 1)
    x0, x1 = max(0, x - r), min(response.shape[1], x + r + 1)
    mask[y0:y1, x0:x1] = False
    side = response[mask]
    if side.size == 0:
        return 0.0
    mu = float(side.mean())
    sigma = float(side.std()) + 1e-9
    return (peak - mu) / sigma


def _rotated_template_bank(
    template: npt.NDArray[np.uint8], angles_deg: tuple[float, ...], *, mirror: bool = False
) -> list[tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8] | None]]:
    """Raw (un-whitened) rotated exemplar crops + masks for the local NCC verify (step 6b).

    This is the *appearance* the verifier correlates against -- the plain crop, NOT the whitened
    MOSSE filter. At 0 deg the crop passes through with ``mask=None``. A non-zero angle warps the
    crop into its axis-aligned bounding box, whose fabricated corners (up to half the template at
    45 deg) are covered by a warped, eroded mask so ``matchTemplate`` scores only real pixels --
    the same corner-honesty ``ncc``'s rotation bank uses (PITFALLS.md 1.6). Built per proposal
    *size* so the exemplar is matched at the instance's detected scale.

    With ``mirror=True`` every one of those variants also yields a horizontally flipped sibling
    (``cv2.flip(..., 1)``), template and mask flipped **together**, after the rotation -- the same
    mechanism ``ncc``'s own ``mirror`` field uses. Flipping the already-eroded mask is valid because
    a flip is a pure reflection on the pixel lattice (it permutes pixels without resampling), so it
    still marks exactly the real template pixels and the corner-honesty invariant carries over.
    """
    h, w = template.shape[:2]
    bank: list[tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8] | None]] = []
    for angle in angles_deg:
        if angle == 0.0:
            warped: npt.NDArray[np.uint8] = template
            mask: npt.NDArray[np.uint8] | None = None
        else:
            rot = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
            cos, sin = abs(float(rot[0, 0])), abs(float(rot[0, 1]))
            new_w, new_h = int(h * sin + w * cos), int(h * cos + w * sin)
            rot[0, 2] += new_w / 2.0 - w / 2.0
            rot[1, 2] += new_h / 2.0 - h / 2.0
            warped = np.ascontiguousarray(
                cv2.warpAffine(
                    template, rot, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=0
                ),
                dtype=np.uint8,
            )
            eroded = cv2.erode(
                cv2.warpAffine(
                    np.full((h, w), 255, np.uint8), rot, (new_w, new_h), flags=cv2.INTER_NEAREST
                ),
                np.ones((3, 3), np.uint8),
            )
            mask = np.asarray(eroded, dtype=np.uint8)
        bank.append((warped, mask))
        if mirror:
            flipped_mask: npt.NDArray[np.uint8] | None = None
            if mask is not None:
                flipped_mask = np.ascontiguousarray(cv2.flip(mask, 1), dtype=np.uint8)
            bank.append((np.ascontiguousarray(cv2.flip(warped, 1), dtype=np.uint8), flipped_mask))
    return bank


def _verify_score(
    gray: npt.NDArray[np.uint8],
    exemplar_crop: npt.NDArray[np.uint8],
    box: BBox,
    angles_deg: tuple[float, ...],
    bank_cache: dict[
        tuple[int, int], list[tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8] | None]]
    ],
    *,
    mirror: bool = False,
) -> float:
    """Local raw ``TM_CCOEFF_NORMED`` of the rotated exemplar around ``box`` -- the 'fine' re-score.

    The whitened filter proposes ``box``; this confirms it with the *raw* normalized correlation
    ``ncc`` uses, but only inside a small window grown from ``box`` (``_VERIFY_MARGIN_FRAC`` of the
    template on each side, so a proposal placed a few pixels off by the pyramid rounding can still
    re-localize). The exemplar is resized to the proposal's detected size so scale is handled by the
    box, and correlated across the rotation bank (plus its mirrored siblings when ``mirror=True``,
    ``MOSSEConfig.mirror`` -- a reflection no rotation bank can reach, the archetype being a
    floor-plan door drawn with the opposite swing hand) so a rotated (VARIED) instance is not
    rejected; the score is the max normalized correlation over the window and the bank, in
    ``[-1, 1]``. Cost is ``O(window * template * angles)`` at the few proposal sites -- the cheap
    half of coarse-to-fine, never ``ncc``'s full-scene sweep. Banks are cached by proposal size (the
    pyramid yields few distinct sizes) so the warps are not rebuilt per peak.
    """
    bw, bh = box.w, box.h
    if bw < _MIN_TEMPLATE_PX or bh < _MIN_TEMPLATE_PX:
        return -1.0
    # Grow the box by a margin on each side so the local correlation can re-localize the proposal.
    mx, my = round(bw * _VERIFY_MARGIN_FRAC), round(bh * _VERIFY_MARGIN_FRAC)
    x0, y0 = max(0, box.x - mx), max(0, box.y - my)
    x1, y1 = min(gray.shape[1], box.x2 + mx), min(gray.shape[0], box.y2 + my)
    win = gray[y0:y1, x0:x1]
    if win.shape[0] < bh or win.shape[1] < bw:
        return -1.0
    bank = bank_cache.get((bw, bh))
    if bank is None:
        resized = np.ascontiguousarray(
            cv2.resize(exemplar_crop, (bw, bh), interpolation=cv2.INTER_AREA), dtype=np.uint8
        )
        bank = _rotated_template_bank(resized, angles_deg, mirror=mirror)
        bank_cache[(bw, bh)] = bank
    best = -1.0
    for tmpl, mask in bank:
        if win.shape[0] < tmpl.shape[0] or win.shape[1] < tmpl.shape[1]:
            continue
        resp = (
            cv2.matchTemplate(win, tmpl, cv2.TM_CCOEFF_NORMED, mask=mask)
            if mask is not None
            else cv2.matchTemplate(win, tmpl, cv2.TM_CCOEFF_NORMED)
        )
        arr = np.asarray(resp, dtype=np.float64)
        finite = arr[np.isfinite(arr)]
        if finite.size:
            best = max(best, float(finite.max()))
    return best


def _empty(
    note: str,
    *,
    threshold: float | None,
    metrics: dict[str, float],
    latency: LatencyBreakdown,
    heatmap: HeatmapPayload | None = None,
) -> SearchResult:
    """Build an honest ``outcome=EMPTY`` result carrying a note that says *why* (METHOD-04c)."""
    return SearchResult(
        method="mosse",
        method_version=_METHOD_VERSION,
        outcome=SearchOutcome.EMPTY,
        matches=(),
        latency=latency,
        threshold_applied=threshold,
        candidates=(),
        diagnostics=Diagnostics(notes=(note,), metrics=metrics, similarity_heatmap=heatmap),
    )


@register_method(
    name="mosse",
    description="MOSSE/ASEF correlation-filter matching via FFT over a scale pyramid.",
    version=_METHOD_VERSION,
    config_model=MOSSEConfig,
)
def search(
    image: npt.NDArray[np.uint8],
    exemplar: ExemplarBox,
    config: BaseModel,
) -> SearchResult:
    """Find every instance of ``exemplar`` in ``image`` with an FFT correlation filter."""
    # The registry's SearchFn protocol types config as BaseModel (it is method-agnostic); the
    # registered config_model guarantees the concrete type. Narrow it once here so the rest of the
    # function is statically a MOSSEConfig, and fail loudly if the contract is ever violated.
    if not isinstance(config, MOSSEConfig):
        raise TypeError(f"mosse.search requires a MOSSEConfig, got {type(config).__name__}")

    height, width = image.shape[:2]
    t_start = perf_counter()

    # 1. Crop the exemplar and guard against a textureless template. A flat crop makes the filter
    #    denominator collapse to the regularizer, so the response is noise; abstain rather than
    #    emit confident garbage (same guard rationale as ncc, METHOD-04c).
    gray: npt.NDArray[np.uint8] = np.ascontiguousarray(
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), dtype=np.uint8
    )
    ex = exemplar.box
    crop = gray[ex.y : ex.y2, ex.x : ex.x2]
    crop_std = float(crop.astype(np.float64).std())
    preprocess_ms = (perf_counter() - t_start) * 1000.0
    if crop_std < _TEMPLATE_STD_FLOOR:
        logger.debug("mosse: textureless exemplar (std={:.3g}); abstaining", crop_std)
        return _empty(
            (
                f"exemplar has no texture for a correlation filter (std={crop_std:.3g} < "
                f"{_TEMPLATE_STD_FLOOR:g}); the MOSSE denominator collapses to the regularizer so "
                "the response is noise, not a match."
            ),
            threshold=None,
            metrics={"crop_std": crop_std},
            latency=LatencyBreakdown(
                preprocess_ms=preprocess_ms, inference_ms=0.0, postprocess_ms=0.0
            ),
        )

    t_after_pre = perf_counter()
    inference_ms = 0.0
    records: list[_LevelPeak] = []
    per_level_counts: dict[str, float] = {}
    heatmap_response: npt.NDArray[np.float64] | None = None
    heatmap_gap = float("inf")

    for scale in config.scales:
        # 2. Build the scale pyramid -- rescale the SCENE, then crop the exemplar region from the
        #    DOWNSCALED scene (identical to ncc: cropping from the resized scene keeps the filter's
        #    geometry aligned with the level and avoids the non-monotone self-match drop that
        #    resizing the template independently causes, PITFALLS.md 1.3).
        level_img: npt.NDArray[np.uint8]
        if scale == 1.0:
            level_img = gray
        else:
            level_img = np.ascontiguousarray(
                cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA),
                dtype=np.uint8,
            )
        lx, ly = round(ex.x * scale), round(ex.y * scale)
        base_tw, base_th = round(ex.w * scale), round(ex.h * scale)
        if base_tw < _MIN_TEMPLATE_PX or base_th < _MIN_TEMPLATE_PX:
            continue
        if level_img.shape[1] < lx + base_tw or level_img.shape[0] < ly + base_th:
            continue
        base_template = np.ascontiguousarray(
            level_img[ly : ly + base_th, lx : lx + base_tw], dtype=np.uint8
        )

        # 3. Synthesize the small MOSSE/ASEF filter BANK from the warped-exemplar set at this level.
        #    The rotation bank folds into a few sharp sub-filters (default 3), one per contiguous
        #    angle sub-range -- the measured sweet spot between one blurry averaged filter and ncc's
        #    7 separate spatial passes.
        t_build = perf_counter()
        kernels = _build_filter_bank(base_template, config)

        # 4. Correlate each sub-filter over the FULL scene with an FFT cross-correlation, normalize
        #    by local window energy -> a cosine-like response in [-1, 1] (the correlation-filter
        #    analogue of TM_CCOEFF_NORMED), and take the per-pixel MAX across the bank. This is the
        #    O(H*W*log(H*W)) step ncc pays spatially, and there are n_angle_groups of them per level
        #    (default 3) instead of ncc's angles-x-levels spatial passes.
        scene_pre = (
            np.log1p(level_img.astype(np.float32))
            if config.log_transform
            else (level_img.astype(np.float32))
        )
        response = _bank_response(scene_pre, kernels, config.energy_floor_frac)
        inference_ms += (perf_counter() - t_build) * 1000.0

        tmpl_h, tmpl_w = kernels[0].shape[:2]
        # Keep the response nearest scale 1.0 for the diagnostics heatmap.
        if abs(scale - 1.0) < heatmap_gap:
            heatmap_gap = abs(scale - 1.0)
            heatmap_response = response

        # 5. Extract peaks per level, standardising cross-level scores first. Like ncc, the response
        #    noise floor varies with template size, so a raw argmax across levels is biased; z-score
        #    against THIS level's own median/MAD (a monotone transform -> peak locations unchanged,
        #    scores comparable across levels). Peaks are picked at 3 sigma.
        finite = response[np.isfinite(response)]
        med = float(np.median(finite)) if finite.size else 0.0
        mad = float(np.median(np.abs(finite - med))) * _MAD_TO_STD + 1e-9
        z_map = np.where(np.isfinite(response), (response - med) / mad, -np.inf)

        level_peaks = peaks.extract_peaks(
            z_map.astype(np.float32),
            strategy=config.peaks,
            template_w=tmpl_w,
            template_h=tmpl_h,
            floor=_PEAK_Z_FLOOR,
            max_peaks=config.max_candidates,
            suppression_radius_frac=config.suppression_radius_frac,
            nms_iou=config.nms_iou,
        )

        # 6. Map peaks to boxes. The response is (H-h+1, W-w+1), top-left anchored -- the value at
        #    (row, col) is a box with top-left (col, row) and the template size, NO centre offset
        #    (PITFALLS.md 1.2). At scale s, divide back into original pixels. Identical to ncc.
        level_count = 0
        for peak in level_peaks:
            bx = max(0, round(peak.x / scale))
            by = max(0, round(peak.y / scale))
            bw = max(1, round(tmpl_w / scale))
            bh = max(1, round(tmpl_h / scale))
            try:
                box = BBox(x=bx, y=by, w=bw, h=bh).clipped_to(width, height)
            except ValueError:
                continue  # a rounded box that fell entirely off-image; drop it
            raw = float(response[peak.y, peak.x])
            records.append(_LevelPeak(box, raw, float(peak.score), scale))
            level_count += 1

        per_level_counts[f"peaks@{scale:g}"] = float(level_count)

    # 6b. Coarse-to-fine verify (default on). The whitened filter localizes but discriminates
    #     weakly in clutter -- it PROPOSES true instances yet scores them alongside clutter, so the
    #     threshold drops them (measured: ~83% of cluttered instances are proposed, but only ~53%
    #     survive without this step). Re-score each proposal with a LOCAL raw TM_CCOEFF_NORMED of
    #     the rotated exemplar -- ncc's discriminative score, but evaluated only in a small window
    #     around each of the few proposals, not over the whole scene -- plus its horizontally
    #     mirrored sibling when config.mirror is set (off by default; a reflection no rotation bank
    #     can reach). That raw score (self-match ~1.0, the anchor the whitened filter lost) replaces
    #     the filter response for the threshold
    #     and the candidate log; the filter's z-score stays the cross-level NMS priority. Off = the
    #     pure filter response (the original shipped behaviour), a control.
    if config.verify and records:
        t_verify = perf_counter()
        bank_cache: dict[
            tuple[int, int], list[tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8] | None]]
        ] = {}
        records = [
            _LevelPeak(
                r.box,
                _verify_score(
                    gray, crop, r.box, config.train_angles_deg, bank_cache, mirror=config.mirror
                ),
                r.z_score,
                r.level,
            )
            for r in records
        ]
        inference_ms += (perf_counter() - t_verify) * 1000.0

    # 7. Calibrate the threshold. repeat-aware (default) reads the score distribution, re-anchored
    #    on the filter's self-response (not 1.0): >=2 distinct near-self locations => near-identical
    #    repeats => strict cut; else transformed instances => permissive self * retain_frac tail.
    #    "fixed" wins whenever the caller pinned config.threshold; the other strategies are the
    #    shared calibration offerings. Every branch returns its reasoning as a diagnostics note.
    raw_scores = [record.raw_score for record in records]
    self_score, exemplar_key = _self_match_score(records, ex)
    if config.threshold is not None:
        # A pinned threshold always wins, whatever calibration is configured.
        calib = calibration.calibrate(
            raw_scores or [0.0], strategy="fixed", fixed_threshold=config.threshold
        )
    elif config.calibration == "repeat-aware":
        # Count DISTINCT near-self locations: the pyramid hits the exemplar's own region at several
        # scales, so the near-self records are NMS-deduplicated before counting -- else every image
        # would look like a near-identical repeat (see _repeat_aware_threshold).
        near = [r for r in records if r.raw_score >= self_score * _REPEAT_NEAR_FRAC]
        near_kept = nms.nms(
            [r.box for r in near], [r.z_score for r in near], iou_threshold=config.nms_iou
        )
        calib = _repeat_aware_threshold(len(near_kept), self_score, retain_frac=config.retain_frac)
    else:
        # One of the shared offerings (self-similarity / ratio / gmm); MyPy narrows the Literal
        # here because repeat-aware was handled just above.
        calib = calibration.calibrate(
            raw_scores or [0.0],
            strategy=config.calibration,
            self_score=self_score,
            retain_frac=config.retain_frac,
            seed=config.seed,
        )
    threshold = calib.threshold

    # 8. Split into matches and sub-threshold candidates. Peaks whose RAW (normalized) score clears
    #    the threshold are cross-level NMS'd (z-score is the comparable priority) into the Matches.
    #    METHOD-12: every accepted peak survives -- there is no single-best short-circuit anywhere.
    ordered = sorted(records, key=lambda r: (-r.z_score, r.box.y, r.box.x))
    accepted = [r for r in ordered if r.raw_score > threshold]
    kept_indices = nms.nms(
        [r.box for r in accepted], [r.z_score for r in accepted], iou_threshold=config.nms_iou
    )
    kept = [accepted[i] for i in kept_indices]
    matches = _build_matches(kept, ex, exemplar_key)

    #    The candidate log (EVAL-08) is the sub-threshold peaks kept WITH RAW SCORES so an offline
    #    threshold sweep can recover the full precision/recall curve. It is deduplicated first (the
    #    pyramid detects one instance at several scales), and any that overlap an accepted match are
    #    dropped, so matches + candidates form ONE clean deduplicated ranked set (fed to AP).
    below = [r for r in ordered if r.raw_score <= threshold]
    below_indices = nms.nms(
        [r.box for r in below], [r.z_score for r in below], iou_threshold=config.nms_iou
    )
    distinct_below = (
        r
        for r in (below[i] for i in below_indices)
        if all(r.box.iou(k.box) < config.nms_iou for k in kept)
    )
    candidates = tuple(
        Candidate(box=r.box, score=r.raw_score)
        for r in list(distinct_below)[: config.max_candidates]
    )

    # 9. Assemble diagnostics and the result.
    postprocess_ms = max(0.0, (perf_counter() - t_after_pre) * 1000.0 - inference_ms)
    latency = LatencyBreakdown(
        preprocess_ms=preprocess_ms, inference_ms=inference_ms, postprocess_ms=postprocess_ms
    )
    heatmap = viz.heatmap_png_b64(heatmap_response) if heatmap_response is not None else None
    # PSR is a diagnostic, so it is computed ONCE for the strongest peak on the representative
    # (nearest scale 1.0) response -- never per candidate, which on a 24-megapixel chipset map would
    # allocate a full-map mask thousands of times and dominate the runtime.
    if heatmap_response is not None:
        py, px = np.unravel_index(int(np.argmax(heatmap_response)), heatmap_response.shape)
        best_psr = _psr(heatmap_response, int(py), int(px))
    else:
        best_psr = 0.0
    metrics: dict[str, float] = {
        "crop_std": crop_std,
        "self_score": self_score,
        "threshold": threshold,
        "best_psr": best_psr,
        "n_candidates": float(len(candidates)),
        "n_matches": float(len(matches)),
        "n_levels_evaluated": float(len(per_level_counts)),
        **per_level_counts,
    }
    notes = (
        f"calibration[{calib.strategy}]: {calib.reason}",
        f"kept {len(matches)} match(es) from {len(records)} peak(s) across "
        f"{len(per_level_counts)} pyramid level(s); threshold {threshold:.4f} on normalized "
        f"filter response; best PSR {best_psr:.1f}.",
    )

    if not matches:
        return _empty(
            (
                "no correlation-filter peak cleared the calibrated threshold "
                f"{threshold:.4f} (best response {max(raw_scores, default=0.0):.4f}); "
                f"{calib.strategy} calibration: {calib.reason}"
            ),
            threshold=threshold,
            metrics=metrics,
            latency=latency,
            heatmap=heatmap,
        )

    return SearchResult(
        method="mosse",
        method_version=_METHOD_VERSION,
        outcome=SearchOutcome.OK,
        matches=matches,
        latency=latency,
        threshold_applied=threshold,
        candidates=candidates,
        diagnostics=Diagnostics(notes=notes, metrics=metrics, similarity_heatmap=heatmap),
    )


def _repeat_aware_threshold(
    n_near_instances: int,
    self_score: float,
    *,
    retain_frac: float,
) -> calibration.CalibrationResult:
    """Distribution-aware cut for the correlation filter: strict on near-identical repeats.

    The correlation filter loses ncc's ~1.0 self-match anchor -- the FILTER is a whitened exemplar,
    so even the exemplar's own region scores a lower, image-dependent ``self_score``. The
    distribution still tells the two cases apart, re-anchored on that self-response: when the object
    repeats near-identically the true instances pile up NEAR the self-response, so two or more
    *distinct* locations at ``>= self_score * _REPEAT_NEAR_FRAC`` mean "near-identical repeats" and
    the cut belongs just below that cluster (``self_score * _REPEAT_STRICT_FRAC``). When only the
    exemplar's own region sits up there, the instances are transformed and score lower, so the cut
    drops to the permissive ``self_score * retain_frac`` tail. ``n_near_instances`` is counted over
    DISTINCT locations (NMS-deduplicated in :func:`search`) because the pyramid detects the
    exemplar's own region several times. Tuned to the distribution *shape*, never to the
    ground-truth boxes; the same rule runs on every dataset (the cross-dataset fairness rule).
    """
    if n_near_instances >= 2:
        threshold = self_score * _REPEAT_STRICT_FRAC
        reason = (
            f"{n_near_instances} distinct locations >= {_REPEAT_NEAR_FRAC:g} x self "
            f"({self_score:.4f}): near-identical repeats -> strict cut self x "
            f"{_REPEAT_STRICT_FRAC:g} = {threshold:.4f} to reject diffuse filter false peaks"
        )
    else:
        threshold = self_score * retain_frac
        reason = (
            f"only the self-response sits near self ({self_score:.4f}): instances look transformed "
            f"-> permissive cut self x retain_frac {retain_frac:g} = {threshold:.4f}"
        )
    return calibration.CalibrationResult(
        threshold=threshold, strategy="repeat-aware", reason=reason
    )


def _self_match_score(
    records: list[_LevelPeak], exemplar: BBox
) -> tuple[float, tuple[int, int, int, int] | None]:
    """Return the exemplar's own filter self-response and a key identifying that peak.

    The exemplar is part of the scene it is searched in, so the filter fires on the exemplar's own
    region -- the best-overlapping peak is that self-response, and its normalized score anchors the
    repeat-aware / self-similarity calibration. Unlike ncc this is NOT ~1.0 (the filter is a
    whitened exemplar, not the exemplar). Falls back to a conservative ``1.0`` when no peak overlaps
    the exemplar (e.g. scale 1.0 was excluded), so calibration still has a self-score to work from.
    """
    best: _LevelPeak | None = None
    best_iou = _EXEMPLAR_IOU
    for record in records:
        overlap = record.box.iou(exemplar)
        if overlap >= best_iou:
            best_iou = overlap
            best = record
    if best is None:
        return 1.0, None
    return best.raw_score, best.box.xyxy


def _build_matches(
    kept: list[_LevelPeak],
    exemplar: BBox,
    exemplar_key: tuple[int, int, int, int] | None,
) -> tuple[Match, ...]:
    """Turn kept peaks into Matches, labelling the exemplar's own region is_exemplar=True.

    The exemplar self-match is a genuine instance and is labelled rather than dropped (understates
    recall) or silently counted as a discovery (overstates the method), per METHOD-04c. Matches are
    returned in the canonical ``(-score, y, x)`` order.
    """
    # Which kept peak is the exemplar's own region: prefer the one whose box matches the key found
    # during calibration, else the highest-IoU box above the overlap floor.
    exemplar_idx: int | None = None
    best_iou = _EXEMPLAR_IOU
    for i, record in enumerate(kept):
        if exemplar_key is not None and record.box.xyxy == exemplar_key:
            exemplar_idx = i
            break
        overlap = record.box.iou(exemplar)
        if overlap >= best_iou:
            best_iou = overlap
            exemplar_idx = i

    ordered = sorted(
        range(len(kept)), key=lambda i: (-kept[i].raw_score, kept[i].box.y, kept[i].box.x)
    )
    return tuple(
        Match(box=kept[i].box, score=kept[i].raw_score, is_exemplar=(i == exemplar_idx))
        for i in ordered
    )
