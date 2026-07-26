"""Method 1 -- normalized cross-correlation template matching (the zero-model baseline).

What it does
------------
Correlate the exemplar crop against the *whole* scene with ``cv2.matchTemplate`` and
``TM_CCOEFF_NORMED``, over an image pyramid **and a rotated-template bank** so instances at
other scales and orientations are found, and turn the response peaks into boxes. No weights,
no training. When the repeated instances are near-identical and near the exemplar's scale this
is genuinely hard to beat, and it is the honest bar every learned method in this project must
clear; the rotation bank + a distribution-aware accept rule (below) also let it recover a
meaningful share of the rotated/rescaled repeats without giving up its fixed-scale strength.

This file is meant to be read top to bottom by an ML practitioner. Readability outranks DRY
(project convention): the numbered steps ``# 1.`` .. ``# 9.`` in :func:`search` match the
headings in ``docs/methods/ncc.md`` one-for-one, and a step may inline a few lines rather
than reach for a shared helper if that reads better standalone.

Pre-processing (exact)
----------------------
- Colour: the BGR scene is converted **once** to single-channel grayscale with
  ``cv2.COLOR_BGR2GRAY``. Grayscale, not per-channel colour correlation, because a
  3-channel ``matchTemplate`` merely *sums* the channel correlations (PITFALLS.md 1.7) and
  the sum is neither more discriminative nor documented.
- dtype/layout: kept **uint8** and made C-contiguous (``np.ascontiguousarray``). It is
  deliberately NOT cast to float32: ``matchTemplate`` gives different numbers for uint8 vs
  float32 input (PITFALLS.md 1.8), so the dtype is part of the method's identity.
- Normalization: **none is applied by hand.** ``TM_CCOEFF_NORMED`` subtracts each window's
  mean and divides by its L2 norm internally, so a separate mean/std normalization would be
  double-counting. This is why there are no ImageNet-style constants anywhere here.
- Search extent: the **full** scene, always. Restricting the search window changes ~73% of
  the returned floats and shifts the peak value (PITFALLS.md 1.8), so a cropped search would
  silently break "same input => identical results". A search restriction, if ever added,
  must join the config (and therefore the config hash).

Post-processing (exact)
-----------------------
- The response map has shape ``(H-h+1, W-w+1)`` and each value is anchored at the
  template's **top-left** corner, not its centre. A response index ``(row, col)`` therefore
  maps to a box ``BBox(x=col, y=row, w=tw, h=th)`` at that level's scale -- no centre
  offset, no +/-1 (PITFALLS.md 1.2). At level scale ``s`` the box in original-image pixels
  is ``x=round(col/s), y=round(row/s), w=round(tw/s), h=round(th/s)``.
- Cross-level comparison is biased: the spurious noise floor of ``TM_CCOEFF_NORMED`` varies
  ~15x with template size (0.577 at 8x8 vs 0.039 at 128x128, PITFALLS.md 1.4), so a naive
  ``argmax`` across pyramid levels favours the smallest template. Each level's map is
  therefore z-scored against its **own** median/MAD before peaks are compared or suppressed
  across levels; the raw score is carried alongside for the threshold and the candidate log.
- Threshold: chosen by the ``repeat-aware`` rule (default), never a hardcoded absolute number.
  It reads the score distribution: if two or more *distinct* locations score near the exemplar's
  own ~1.0 self-match the object is repeating near-identically, so the cut sits just below that
  cluster (``self x 0.85``) -- high enough to reject the moderate (~0.5-0.76) false peaks a
  rotated template throws on structured backgrounds, which a single low fixed cut cannot separate
  because they outscore genuine *transformed* instances elsewhere. When only the exemplar's own
  region sits up there, the instances are transformed and score lower, so the cut drops to the
  permissive ``self x retain_frac`` (0.45) tail. The other ``common.calibration`` strategies
  (``self-similarity``/``ratio``/``gmm``) remain selectable. Tuned to the distribution's *shape*,
  never to ground-truth boxes; the same rule runs on every dataset.
- Suppression: cross-level greedy IoU NMS over the accepted matches, prioritised by the
  cross-level-comparable z-score.
- Candidate log (EVAL-08): the sub-threshold peaks, **deduplicated** by the same cross-level NMS
  and with any that overlap an accepted match dropped, so matches + candidates form one clean
  ranked detection set. Without this dedup a single instance -- detected at many (scale, angle)
  pairs by the pyramid x rotation bank -- would enter the log dozens of times and each duplicate
  would score as a false positive in the AP sweep, badly understating AP.

Known failure modes
--------------------
- **Textureless crop.** A flat exemplar makes ``TM_CCOEFF_NORMED`` return ``1.0`` at *every*
  pixel on OpenCV 4.x (``0.0`` on 5.x) -- measured, undocumented, and never a NaN
  (PITFALLS.md 1.1). Step 1 guards on the crop's own std and abstains with ``outcome=EMPTY``
  rather than emitting a wall of confident false positives.
- **Rotation/scale beyond the configured banks.** The default bank covers +/-35 deg and
  scales 0.75-1.3; instances rotated or scaled past that are missed, and even inside the bank a
  rotated-and-rescaled instance whose resampled correlation falls below ``self x retain_frac`` is
  missed. Recall in the scale/pose regimes is genuinely partial (VARIED ~0.36) -- an inherent
  ceiling of raw-intensity correlation, which is exactly where ``dino-dense`` and ``sparse-geo``
  win.
- **Lighting/pose change.** NCC correlates raw intensities, so an instance under different
  lighting scores low even when a human sees the same object.
- **Cross-level noise-floor bias** (mitigated by the per-level z-score above; called out so
  a future editor does not "simplify" it away).

ROBUSTNESS BACKLOG
------------------
Deferred deliberately (mirrored verbatim in ``docs/methods/ncc.md`` and
``docs/ROBUSTNESS-BACKLOG.md``); none is built in this phase:

- **FFT-based correlation for large templates.** The spatial ``matchTemplate`` is O(H*W*h*w);
  a single full-scene FFT cross-correlation is O(H*W*log(H*W)) and wins decisively once the
  template is large.
- **Log-polar / Fourier-Mellin registration** for joint rotation+scale invariance in one
  correlation, replacing the brute-force rotated-template x pyramid bank.
- **Discriminative correlation filters (MOSSE/KCF)** trained on the single exemplar crop, so
  the filter learns to suppress background instead of correlating raw pixels.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from time import perf_counter
from typing import Literal

import cv2
import numpy as np
import numpy.typing as npt
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

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
# Below this the crop is textureless and TM_CCOEFF_NORMED is undefined (PITFALLS.md 1.1). A
# uniform crop has std exactly 0.0; the epsilon catches float dust without swallowing a
# genuinely low-but-nonzero texture.
_TEMPLATE_STD_FLOOR = 1e-6
# matchTemplate raises if the template exceeds the level image, and a sub-8px template is
# statistically worthless (PITFALLS.md 1.4/1.5). Levels producing one are skipped.
_MIN_TEMPLATE_PX = 8
# Peaks are picked on the per-level z-score, so the floor is in standard deviations above the
# level's own noise -- 3 sigma. This is what makes cross-level comparison honest (1.4).
_PEAK_Z_FLOOR = 3.0
_MAD_TO_STD = 1.4826  # median-absolute-deviation -> std, for a Gaussian
# A peak whose box overlaps the exemplar by at least this is the exemplar's own self-match,
# labelled is_exemplar rather than dropped or double-counted (METHOD-04c).
_EXEMPLAR_IOU = 0.5
# repeat-aware calibration. A match at >= self_score * _REPEAT_NEAR_FRAC counts as "near the
# self-match"; two or more of those means the object is repeating near-identically (the chipset
# / fixed-scale case), where the honest cut sits just below that cluster at self_score *
# _REPEAT_STRICT_FRAC -- high enough to reject the moderate (~0.5-0.76) false peaks a rotated
# template throws on structured backgrounds, which no fixed low fraction can separate because
# those false peaks outscore genuine transformed instances in the varied/cluttered regimes.
_REPEAT_NEAR_FRAC = 0.9
_REPEAT_STRICT_FRAC = 0.85


class NCCConfig(BaseModel):
    """Frozen config for :func:`search`; its JSON Schema drives the UI form (one source).

    Every field carries a ``description`` because that text becomes the form's help string --
    this is the single place it is written, so it must be written here.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    scales: tuple[float, ...] = Field(
        default=(0.75, 0.875, 1.0, 1.15, 1.3),
        description=(
            "Pyramid scale factors. The SCENE is resized by each factor and the template is "
            "cropped from that resized scene, which keeps the self-match at 1.0 (see docs)."
        ),
    )
    angles_deg: tuple[float, ...] = Field(
        default=(-35.0, -23.3, -11.7, 0.0, 11.7, 23.3, 35.0),
        description=(
            "Rotation bank in degrees. Default is a 7-step bank over +/-35 deg (~11.7 deg "
            "spacing): raw-intensity correlation loses a rotated instance within ~10-15 deg, so "
            "a bank this dense is what lets NCC recover the rotated repeats in the scale/pose "
            "regimes. It is a constant-factor cost (levels x angles correlations); a caller who "
            "knows the scene is axis-aligned can set (0.0,) to skip it. 7 angles measured best: "
            "9 over-samples (adds false peaks, no recall gain), 5 leaves gaps."
        ),
    )
    threshold: float | None = Field(
        default=None,
        description="Fixed accept threshold on the raw NCC score. None => use the calibrator.",
    )
    calibration: Literal["fixed", "self-similarity", "ratio", "gmm", "repeat-aware"] = Field(
        default="repeat-aware",
        description=(
            "How the accept threshold is chosen when `threshold` is None. repeat-aware (default) "
            "reads the score distribution: if several matches cluster near the exemplar's own "
            "~1.0 self-match the object is repeating near-identically, so it cuts just below that "
            "cluster (strict, rejects the rotated-template false peaks); otherwise the instances "
            "are transformed and score lower, so it cuts at self_score * retain_frac to keep that "
            "tail. self-similarity is the plain fixed-fraction cut; ratio/gmm are the controls."
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
        default=0.45,
        gt=0.0,
        le=1.0,
        description=(
            "The permissive self-relative accept fraction: a match is kept above "
            "self_score * retain_frac. Used directly by self-similarity, and as the "
            "transformed-instance floor by repeat-aware. 0.45 because a rotated-and-rescaled "
            "true instance correlates to roughly 0.4-0.6 of the exemplar's self-match once "
            "resampling has degraded it; 0.45 sits on the broad F1 plateau across the varied/"
            "cluttered regimes and is NOT fit to the ground-truth boxes (see docs/methods/ncc.md)."
        ),
    )


@dataclass(frozen=True)
class _LevelPeak:
    """One extracted peak carried through post-processing with everything the split needs.

    ``raw_score`` is the untouched ``TM_CCOEFF_NORMED`` value (what the threshold and the
    candidate log use); ``z_score`` is the per-level-standardised value (what cross-level
    ranking and NMS use, since raw scores are not comparable across template sizes).
    """

    box: BBox
    raw_score: float
    z_score: float
    level: float
    angle: float


def _rotated_bank(
    template: npt.NDArray[np.uint8],
    angles_deg: tuple[float, ...],
) -> Iterator[tuple[float, npt.NDArray[np.uint8], npt.NDArray[np.uint8] | None]]:
    """Yield ``(angle, template, mask)`` for each requested angle.

    At 0 degrees the template is passed through untouched with ``mask=None``. For a non-zero
    angle the crop is warped into its axis-aligned bounding box, whose corners are then
    fabricated constant pixels -- up to *half* the template at 45 degrees (PITFALLS.md 1.6).
    Correlating those zeros against uniform scene regions inflates the score, so a matching
    **mask** is warped alongside and eroded to kill the interpolated fringe, and passed to
    ``matchTemplate`` so only the real pixels count. (Chosen over inscribing an axis-aligned
    rectangle, which would throw away real template pixels near the corners.)
    """
    h, w = template.shape[:2]
    for angle in angles_deg:
        if angle == 0.0:
            yield 0.0, template, None
            continue
        rot = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
        cos, sin = abs(float(rot[0, 0])), abs(float(rot[0, 1]))
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)
        rot[0, 2] += new_w / 2.0 - w / 2.0
        rot[1, 2] += new_h / 2.0 - h / 2.0
        warped = cv2.warpAffine(
            template, rot, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=0
        )
        mask = cv2.warpAffine(
            np.full((h, w), 255, np.uint8), rot, (new_w, new_h), flags=cv2.INTER_NEAREST
        )
        mask = cv2.erode(mask, np.ones((3, 3), np.uint8))
        yield angle, np.ascontiguousarray(warped, dtype=np.uint8), np.asarray(mask, dtype=np.uint8)


def _correlate(
    scene: npt.NDArray[np.uint8],
    template: npt.NDArray[np.uint8],
    mask: npt.NDArray[np.uint8] | None,
) -> npt.NDArray[np.float64]:
    """Full-scene ``TM_CCOEFF_NORMED``, with NaN replaced by -inf so it cannot win an argmax.

    Any non-finite value (a masked correlation can produce one, PITFALLS.md 1.6) is mapped to
    ``-inf`` -- never a peak -- before the map leaves this function.
    """
    if mask is None:
        resp = cv2.matchTemplate(scene, template, cv2.TM_CCOEFF_NORMED)
    else:
        resp = cv2.matchTemplate(scene, template, cv2.TM_CCOEFF_NORMED, mask=mask)
    arr = np.asarray(resp, dtype=np.float64)
    return np.where(np.isfinite(arr), arr, -np.inf)


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
        method="ncc",
        method_version=_METHOD_VERSION,
        outcome=SearchOutcome.EMPTY,
        matches=(),
        latency=latency,
        threshold_applied=threshold,
        candidates=(),
        diagnostics=Diagnostics(notes=(note,), metrics=metrics, similarity_heatmap=heatmap),
    )


@register_method(
    name="ncc",
    description="Normalized cross-correlation template matching over a scale pyramid.",
    version=_METHOD_VERSION,
    config_model=NCCConfig,
)
def search(
    image: npt.NDArray[np.uint8],
    exemplar: ExemplarBox,
    config: BaseModel,
) -> SearchResult:
    """Find every instance of ``exemplar`` in ``image`` by normalized cross-correlation."""
    # The registry's SearchFn protocol types config as BaseModel (it is method-agnostic); the
    # registered config_model guarantees the concrete type. Narrow it once here so the rest of
    # the function is statically an NCCConfig, and fail loudly if the contract is ever violated.
    if not isinstance(config, NCCConfig):
        raise TypeError(f"ncc.search requires an NCCConfig, got {type(config).__name__}")

    height, width = image.shape[:2]
    t_start = perf_counter()

    # 1. Crop the exemplar and guard against a textureless template.
    #    Compute the crop's std FIRST. A flat crop makes TM_CCOEFF_NORMED return a CONSTANT map
    #    (1.0 everywhere on OpenCV 4.10, 0.0 on 4.13/5.x) -- measured, undocumented, never a NaN
    #    (PITFALLS.md 1.1). Without this guard a textureless crop becomes a wall of confident
    #    false positives (4.10) or a silent nothing (4.13+), so abstaining here is mandatory.
    gray: npt.NDArray[np.uint8] = np.ascontiguousarray(
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), dtype=np.uint8
    )
    ex = exemplar.box
    crop = gray[ex.y : ex.y2, ex.x : ex.x2]
    crop_std = float(crop.astype(np.float64).std())
    preprocess_ms = (perf_counter() - t_start) * 1000.0
    if crop_std < _TEMPLATE_STD_FLOOR:
        logger.debug("ncc: textureless exemplar (std={:.3g}); abstaining", crop_std)
        return _empty(
            (
                f"exemplar has no texture for NCC (std={crop_std:.3g} < {_TEMPLATE_STD_FLOOR:g}); "
                "TM_CCOEFF_NORMED is undefined for a zero-variance template and returns a "
                "constant map, so a match everywhere would be a false-positive wall."
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
        # 2. Build the scale pyramid -- rescale the SCENE, then crop the template from the
        #    DOWNSCALED scene. This reverses the intuitive "resize the template" approach, and
        #    the reversal is measured: resizing the template independently drops the exemplar's
        #    own self-match from 1.0000 to 0.3071 NON-monotonically (PITFALLS.md 1.3), so it
        #    cannot be fixed with a per-level offset. Cropping from the already-resized scene
        #    keeps the self-match at 1.0000. It looks wrong; it is right.
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

        level_count = 0
        # 3. Build the rotated-template bank (default: 7 angles over +/-35 deg). This is what
        #    recovers rotated repeats; a caller who knows the scene is axis-aligned sets (0.0,).
        for angle, template, mask in _rotated_bank(base_template, config.angles_deg):
            tmpl_h, tmpl_w = template.shape[:2]
            if level_img.shape[0] < tmpl_h or level_img.shape[1] < tmpl_w:
                continue

            # 4. Correlate over the FULL scene at this level (never a cropped search region).
            t_corr = perf_counter()
            response = _correlate(level_img, template, mask)
            inference_ms += (perf_counter() - t_corr) * 1000.0

            # Keep the raw response nearest scale 1.0 (unrotated) for the diagnostics heatmap.
            if angle == 0.0 and abs(scale - 1.0) < heatmap_gap:
                heatmap_gap = abs(scale - 1.0)
                heatmap_response = response

            # 5. Extract peaks per level, standardising cross-level scores first. The spurious
            #    noise floor varies ~15x with template size (PITFALLS.md 1.4), so a raw argmax
            #    across levels favours the smallest template. Z-score against THIS level's own
            #    median/MAD; because that transform is monotone, peak LOCATIONS are unchanged,
            #    but the returned z is now comparable across levels. Peaks are picked at 3 sigma.
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

            # 6. Map peaks to boxes. The response is (H-h+1, W-w+1), top-left anchored -- the
            #    value at (row, col) is a box with top-left (col, row) and the template size, NO
            #    centre offset (PITFALLS.md 1.2). At scale s, divide back into original pixels.
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
                records.append(_LevelPeak(box, raw, float(peak.score), scale, angle))
                level_count += 1

        per_level_counts[f"peaks@{scale:g}"] = float(level_count)

    # 7. Calibrate the threshold. repeat-aware (default) reads the score distribution: near-
    #    identical repeats cluster at the self-match and get a strict cut just below it, while a
    #    transformed set (no such cluster) gets the permissive self_score * retain_frac cut.
    #    "fixed" is used whenever the caller pinned config.threshold; the other strategies are the
    #    shared calibration offerings. Every branch returns its reasoning as a diagnostics note.
    raw_scores = [record.raw_score for record in records]
    self_score, exemplar_key = _self_match_score(records, ex)
    if config.threshold is not None:
        # A pinned threshold always wins, whatever calibration is configured.
        calib = calibration.calibrate(
            raw_scores or [0.0], strategy="fixed", fixed_threshold=config.threshold
        )
    elif config.calibration == "repeat-aware":
        # Count DISTINCT near-self locations: the pyramid x rotation bank hits the exemplar's own
        # region many times, so the near-self records are NMS-deduplicated before counting -- else
        # every image would look like a near-identical repeat (see _repeat_aware_threshold).
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

    # 8. Split into matches and sub-threshold candidates. Peaks whose RAW score clears the
    #    threshold are cross-level NMS'd (z-score is the comparable priority) into the Matches.
    #    METHOD-12: every accepted peak survives -- there is no single-best short-circuit anywhere.
    ordered = sorted(records, key=lambda r: (-r.z_score, r.box.y, r.box.x))
    accepted = [r for r in ordered if r.raw_score > threshold]
    kept_indices = nms.nms(
        [r.box for r in accepted], [r.z_score for r in accepted], iou_threshold=config.nms_iou
    )
    kept = [accepted[i] for i in kept_indices]
    matches = _build_matches(kept, ex, exemplar_key)

    #    The candidate log (EVAL-08) is the sub-threshold peaks kept WITH RAW SCORES so an offline
    #    threshold sweep can recover the full precision/recall curve. It is *deduplicated* first --
    #    the pyramid x rotation bank detects one instance at many (scale, angle) pairs, so without
    #    NMS a single location would enter the log dozens of times and each duplicate would score
    #    as a false positive in the AP sweep, understating AP badly. Sub-threshold peaks are cross-
    #    level NMS'd, and any that overlap an accepted match are dropped, so matches + candidates
    #    form ONE clean deduplicated ranked set (which is what the benchmark feeds to AP).
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
    metrics: dict[str, float] = {
        "crop_std": crop_std,
        "self_score": self_score,
        "threshold": threshold,
        "n_candidates": float(len(candidates)),
        "n_matches": float(len(matches)),
        "n_levels_evaluated": float(len(per_level_counts)),
        **per_level_counts,
    }
    notes = (
        f"calibration[{calib.strategy}]: {calib.reason}",
        f"kept {len(matches)} match(es) from {len(records)} peak(s) across "
        f"{len(per_level_counts)} pyramid level(s); threshold {threshold:.4f} on raw NCC.",
    )

    if not matches:
        return _empty(
            (
                "no NCC peak cleared the calibrated threshold "
                f"{threshold:.4f} (best raw {max(raw_scores, default=0.0):.4f}); "
                f"{calib.strategy} calibration: {calib.reason}"
            ),
            threshold=threshold,
            metrics=metrics,
            latency=latency,
            heatmap=heatmap,
        )

    return SearchResult(
        method="ncc",
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
    """Distribution-aware NCC cut: strict when the object repeats near-identically, else permissive.

    The rotated-template bank is what recovers rotated instances, but on a scene of near-identical
    axis-aligned repeats (the chipset regime) those rotated templates also throw *moderate* false
    peaks -- measured at raw 0.5-0.76, higher than genuine transformed instances score elsewhere,
    so no single low fixed cut can separate the two across regimes. The distribution tells them
    apart: when the true instances are near-identical they pile up AT the self-match (~1.0), so two
    or more *distinct* locations at ``>= self_score * _REPEAT_NEAR_FRAC`` mean "near-identical
    repeats" and the cut belongs just below that cluster (``self_score * _REPEAT_STRICT_FRAC``),
    rejecting the moderate false peaks. When only the exemplar's own self-match sits up there, the
    instances are transformed and score lower, so the cut drops to the permissive ``self_score *
    retain_frac`` tail. ``n_near_instances`` is counted over DISTINCT locations (the near-self
    records deduplicated by NMS in :func:`search`), because the pyramid x rotation bank detects the
    exemplar's own region many times over -- counting raw peaks would call every image a repeat.
    This is tuned to the *shape* of the score distribution, never to the ground-truth boxes, and
    the same rule runs on every dataset (the cross-dataset fairness rule).
    """
    if n_near_instances >= 2:
        threshold = self_score * _REPEAT_STRICT_FRAC
        reason = (
            f"{n_near_instances} distinct locations >= {_REPEAT_NEAR_FRAC:g} x self "
            f"({self_score:.4f}): near-identical repeats -> strict cut self x "
            f"{_REPEAT_STRICT_FRAC:g} = {threshold:.4f} to reject rotated-template false peaks"
        )
    else:
        threshold = self_score * retain_frac
        reason = (
            f"only the self-match sits near self ({self_score:.4f}): instances look transformed "
            f"-> permissive cut self x retain_frac {retain_frac:g} = {threshold:.4f}"
        )
    return calibration.CalibrationResult(
        threshold=threshold, strategy="repeat-aware", reason=reason
    )


def _self_match_score(
    records: list[_LevelPeak], exemplar: BBox
) -> tuple[float, tuple[int, int, int, int] | None]:
    """Return the exemplar's own self-match raw score and a key identifying that peak.

    The exemplar is part of the scene it is searched in, so at scale 1.0 it self-correlates to
    ~1.0 (PITFALLS.md 1.3). The best-overlapping peak is that self-match; its raw score anchors
    self-similarity calibration. Falls back to a theoretical ``1.0`` when no peak overlaps the
    exemplar (e.g. scale 1.0 was excluded), so calibration still has a self-score to work from.
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

    The exemplar self-match is a genuine instance and is labelled rather than dropped
    (understates recall) or silently counted as a discovery (overstates the method), per
    METHOD-04c. Matches are returned in the canonical ``(-score, y, x)`` order.
    """
    # Which kept peak is the exemplar's own region: prefer the one whose box matches the key
    # found during calibration, else the highest-IoU box above the overlap floor.
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
