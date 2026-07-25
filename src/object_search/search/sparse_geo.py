"""Method 2 -- sparse keypoint matching plus geometric verification (multi-instance).

What it does
------------
Detect local keypoints on the exemplar crop and on the whole scene, match crop keypoints to
scene keypoints **many-to-many**, then recover **many** geometric models -- one per instance --
by clustering the matches in pose space (generalized Hough) and verifying each cluster with its
own RANSAC-fitted 4-DoF similarity. This is Lowe's original multi-object recognition pipeline
(IJCV 2004 §7.3), and it is aimed squarely at the case NCC struggles with: instances that are
the same object but sit at different rotations or scales.

This file is meant to be read top to bottom by an ML practitioner. Readability outranks DRY
(project convention): the numbered steps ``# 1.`` .. ``# 9.`` in :func:`search` match the
headings in ``docs/methods/sparse-geo.md`` one-for-one, and a step may inline a few lines
rather than reach for a shared helper if that reads better standalone.

The single most load-bearing design choice
-------------------------------------------
**The standard Lowe ratio test is DISABLED.** Lowe's ratio test (keep a match only if the best
neighbour is much closer than the *second* best) exists specifically to reject descriptors that
have several equally-good matches -- and several equally-good matches is the exact signature of
a repeated instance. Every crop keypoint on a repeated object matches one scene keypoint per
instance at near-equal distance, so the standard ratio test would discard precisely the
correspondences this method needs. We therefore take the top-``k`` scene neighbours
**unconditionally** (``k`` is an explicit ceiling on findable instances) and offer only the
optional **k+1 ratio** -- comparing the k-th neighbour to the (k+1)-th -- which still rejects a
descriptor that is non-discriminative against the *whole* image while keeping up to ``k``
repeats. When the k-th neighbour is still a strong match, instances were probably truncated at
the ceiling; that is surfaced as ``k_ceiling_hit`` in diagnostics.

Pre-processing (exact)
----------------------
- Colour: the BGR scene is converted **once** to single-channel grayscale with
  ``cv2.COLOR_BGR2GRAY``. All three classical detectors operate on intensity, not colour.
- dtype/layout: kept **uint8**, C-contiguous. No mean/std normalization is applied -- SIFT,
  AKAZE and ORB each normalize their own descriptors internally, and inventing a second
  normalization here would only desynchronise the crop and scene descriptors.
- The exemplar crop is ``gray[y:y2, x:x2]``; keypoints detected on it are shifted by the crop
  origin ``(x, y)`` so every coordinate in this module lives in **scene** pixels.
- Backend defaults to **SIFT**, not ORB. Research measured ORB yielding ~1 keypoint where SIFT
  yields 83 on the same 64x64 crop; on the small crops this project deals in, ORB routinely
  falls below the vote floor and makes the method look broken when the detector is the problem.
- **The descriptor distance metric is a property of the backend, never a config field.** SIFT
  and AKAZE (configured for its float KAZE descriptor) are L2; ORB's binary descriptor is
  Hamming. Getting this wrong yields garbage matches that still *look* like matches, so it is
  chosen from the backend and cannot be set to the wrong value from the UI.

Post-processing (exact)
-----------------------
- Votes are cast in 4-DoF pose space ``(centre_x, centre_y, log_scale, theta)`` with **soft
  binning** -- each vote lands in the 2 nearest bins per dimension (16 bins in 4-DoF, Lowe's
  boundary fix) -- and **theta wraps circularly** so a vote near 0/360 degrees reaches the
  adjacent bin rather than the opposite end of the histogram.
- Bin widths are Lowe's verified §7.3 values: **30 degrees** orientation, **factor 2** scale,
  **0.25 x the max projected crop dimension** location. The location bin width is
  **scale-dependent**, which is why votes live in a **dict keyed by the bin tuple**, not a
  dense array.
- Each peak (>= ``min_votes`` weight, de-duplicated against its 3^4 neighbourhood) is verified
  by a **NumPy** RANSAC that fits a 4-DoF similarity, seeded from
  ``np.random.default_rng(config.seed)`` -- **not** ``cv2.setRNGSeed``, which has no effect on
  OpenCV RANSAC. A peak is accepted when its inliers reach ``min_inliers``.
- Degeneracy rejection uses **scale plausibility** and **mirror rejection** (a negative
  determinant of the fitted 2x2 linear part). Shear and aspect distortion are deliberately NOT
  tested: a 4-DoF similarity has neither by construction, so those tests are vacuous.
- Multiple distinct models are returned (METHOD-12); there is no single-best short-circuit.

Known failure modes
--------------------
- **Textureless / low-keypoint crop.** Below ``min_exemplar_keypoints`` the crop cannot support
  matching; the method returns ``outcome=EMPTY`` with a diagnostic note rather than a silent
  empty result (METHOD-04c).
- **Small near-identical instances (the NCC crossover).** When instances are small and nearly
  identical, almost every tentative match is wrong and Hough's discriminative power is
  insufficient -- and that is exactly the regime where Method 1 (NCC) is strongest. This is an
  **expected finding, not a bug**; it is a large part of why four methods exist, and the Phase 8
  benchmark should demonstrate the crossover rather than hide it.
- **ORB on tiny crops.** ORB's keypoint yield collapses on small crops; SIFT is the default for
  this reason.

ROBUSTNESS BACKLOG
------------------
Deferred deliberately (mirrored in ``docs/methods/sparse-geo.md``); none is built in this phase:

- **Multi-model fitting (J-linkage / T-linkage)** as a third decomposition strategy alongside
  Hough voting and sequential RANSAC.
- **DISK / ALIKED backends** -- additional learned detectors and the permissive-licence escape
  from SuperPoint's non-commercial terms.
- **Post-hoc orientation/scale assignment for frameless keypoints** via local gradient
  histograms, which would unlock ``single-4dof`` voting for a learned backend.
- **LoFTR / RoMa dense matching** with correspondence-field clustering for low-texture objects
  (a research spike -- the ONNX export is awkward).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
import numpy.typing as npt
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

# -- Tunables that are properties of the METHOD, not of a query, so they are module -------
# -- constants rather than config fields. Each is justified from Lowe IJCV 2004 §7.3. -----

_METHOD_VERSION = "1.0.0"
# Lowe's verified §7.3 pose-space bin widths.
_THETA_BIN_DEG = 30.0  # orientation bin width in degrees
_N_THETA_BINS = 12  # 360 / 30 -- theta is binned modulo this many bins (circular wrap)
_SCALE_BIN_FACTOR = 2.0  # scale bin spans a factor of 2 -> width log(2) in log-scale space
_LOCATION_BIN_FRAC = 0.25  # location bin width = 0.25 x max projected crop dimension
# When the k-th neighbour is at least this fraction of the (k+1)-th distance, the k cutoff is
# not discriminating the instances from the tail -- either the k+1 ratio drops the keypoint, or
# (always) it is counted towards k_ceiling_hit so a truncated instance set is visible.
_K_CEILING_RATIO = 0.9
# A verified peak whose mapped box overlaps the exemplar by at least this IoU is the exemplar's
# own self-match, labelled is_exemplar rather than dropped or double-counted (METHOD-04c).
_EXEMPLAR_IOU = 0.5


class SparseGeoConfig(BaseModel):
    """Frozen config for :func:`search`; its JSON Schema drives the UI form (one source).

    Every field carries a ``description`` because that text becomes the form's help string --
    this is the single place it is written, so it must be written here.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    backend: Literal["sift", "akaze", "orb"] = Field(
        default="sift",
        description=(
            "Keypoint detector/descriptor. SIFT is the default: ORB yields ~1 keypoint where "
            "SIFT yields 83 on a 64px crop, so ORB falls below the vote floor on small crops. "
            "The descriptor DISTANCE METRIC is fixed by the backend (SIFT/AKAZE float L2, ORB "
            "binary Hamming) and is deliberately NOT a separate field."
        ),
    )
    k: int = Field(
        default=6,
        ge=1,
        description=(
            "Top-k scene neighbours kept per crop keypoint, UNCONDITIONALLY (no standard ratio "
            "test). k is an explicit ceiling on the number of instances findable per keypoint."
        ),
    )
    use_kplus1_ratio: bool = Field(
        default=False,
        description=(
            "Enable the ONLY ratio test available: compare the k-th neighbour distance to the "
            "(k+1)-th and drop the crop keypoint when they are within kplus1_ratio. This rejects "
            "descriptors non-discriminative against the whole image while keeping up to k repeats."
        ),
    )
    kplus1_ratio: float = Field(
        default=0.9,
        gt=0.0,
        le=1.0,
        description="Drop a crop keypoint when dist(k-th) >= kplus1_ratio * dist((k+1)-th).",
    )
    voting_mode: Literal["single-4dof", "translation-2dof", "pairwise-4dof"] = Field(
        default="single-4dof",
        description=(
            "How a correspondence becomes a pose vote. single-4dof needs keypoints with a frame "
            "(scale+orientation) and RAISES on a frameless backend; translation-2dof votes in "
            "(dx,dy) only; pairwise-4dof fits a similarity from each pair of correspondences."
        ),
    )
    decomposition: Literal["hough", "sequential-ransac"] = Field(
        default="hough",
        description=(
            "Cluster-into-instances strategy. hough votes in pose space then verifies each peak; "
            "sequential-ransac fits the dominant model, removes its inliers, and repeats."
        ),
    )
    min_votes: int = Field(
        default=3,
        ge=1,
        description="Minimum accumulated vote weight in a bin to hypothesize an instance cluster.",
    )
    min_inliers: int = Field(
        default=5,
        ge=2,
        description="Minimum RANSAC inliers to accept a verified peak as a real instance.",
    )
    pairwise_cap: int = Field(
        default=20000,
        ge=1,
        description="Cap on sampled correspondence pairs for pairwise-4dof (it is O(n^2)).",
    )
    min_exemplar_keypoints: int = Field(
        default=20,
        ge=1,
        description=(
            "Below this many exemplar keypoints the crop lacks texture for this method; search "
            "returns outcome=EMPTY WITH a diagnostic note, never a silent empty result."
        ),
    )
    ransac_iters: int = Field(
        default=200,
        ge=1,
        description="RANSAC iterations per peak (2-point minimal samples).",
    )
    ransac_thresh_px: float = Field(
        default=5.0,
        gt=0.0,
        description="Inlier reprojection-error threshold in scene pixels.",
    )
    min_scale: float = Field(
        default=0.2,
        gt=0.0,
        description="Reject a fitted model whose scale (relative to the exemplar) is below this.",
    )
    max_scale: float = Field(
        default=5.0,
        gt=0.0,
        description="Reject a fitted model whose scale (relative to the exemplar) exceeds this.",
    )
    seed: int = Field(
        default=0,
        ge=0,
        description=(
            "Seed for np.random.default_rng, which drives per-peak RANSAC sampling and pairwise "
            "pair sampling. This is the REAL seed: cv2.setRNGSeed does not affect OpenCV RANSAC."
        ),
    )


# ------------------------------------------------------------------- backend abstraction


@dataclass(frozen=True)
class _Backend:
    """A detector/descriptor behind one interface.

    ``metric`` and ``has_frame`` are the only two things the rest of the module needs to know
    about a backend: which descriptor distance to use, and whether its keypoints carry a
    geometric frame (scale + orientation) -- the latter is what makes ``single-4dof`` valid.
    """

    name: str
    metric: Literal["l2", "hamming"]
    has_frame: bool
    detector: cv2.Feature2D


def _make_backend(name: Literal["sift", "akaze", "orb"]) -> _Backend:
    """Construct the requested classical backend, fixing its distance metric.

    AKAZE is configured for its **float KAZE descriptor** (``DESCRIPTOR_KAZE``) rather than the
    default binary MLDB, so that -- as the method contract states -- SIFT and AKAZE are both L2
    and only ORB is Hamming. All three classical detectors produce keypoints with a full frame
    (``size`` and ``angle``), so ``has_frame`` is True; a frameless backend (SuperPoint) would
    set it False and make ``single-4dof`` voting raise.
    """
    # The cv2 type stubs omit the detector FACTORY functions (SIFT_create et al.) while typing
    # the Feature2D instances they return, so these three calls need an attr-defined ignore.
    if name == "sift":
        return _Backend("sift", "l2", True, cv2.SIFT_create())  # type: ignore[attr-defined]
    if name == "akaze":
        detector = cv2.AKAZE_create(  # type: ignore[attr-defined]
            descriptor_type=cv2.AKAZE_DESCRIPTOR_KAZE
        )
        return _Backend("akaze", "l2", True, detector)
    if name == "orb":
        return _Backend("orb", "hamming", True, cv2.ORB_create())  # type: ignore[attr-defined]
    raise ValueError(f"unknown backend {name!r}")  # unreachable via the Literal, defensive


@dataclass(frozen=True)
class _Keypoints:
    """Detected keypoints in **scene** coordinates plus their descriptors.

    ``scale`` and ``angle`` are ``None`` for a frameless backend. Keeping them optional is what
    lets the voting layer validate the mode against the backend rather than crashing halfway.
    """

    xy: npt.NDArray[np.float64]  # (n, 2) scene pixels
    scale: npt.NDArray[np.float64] | None  # (n,) keypoint size, or None if frameless
    angle: npt.NDArray[np.float64] | None  # (n,) degrees, or None if frameless
    descriptors: npt.NDArray[np.generic]  # (n, D) float32 (L2) or uint8 (Hamming)

    @property
    def count(self) -> int:
        return int(self.xy.shape[0])


def _abstain_note(backend: str, n_keypoints: int, minimum: int) -> str:
    """The diagnostic the low-keypoint guard emits (METHOD-04c) -- why the crop was abstained on.

    A crop with too few keypoints cannot support matching-plus-geometry, but the cause is
    *insufficient texture*, not "found nothing". Returning ``outcome=EMPTY`` WITH this note keeps
    the abstention legible instead of looking like a bug or a real negative result.
    """
    return (
        f"exemplar yielded only {n_keypoints} {backend} keypoint(s) (< {minimum} required); "
        f"too little texture for sparse-geo -- abstaining rather than guessing"
    )


def _detect(
    gray: npt.NDArray[np.uint8],
    backend: _Backend,
    origin_xy: tuple[int, int] = (0, 0),
) -> _Keypoints:
    """Detect and describe with ``backend``, shifting coordinates by ``origin_xy``.

    ``origin_xy`` is the crop's top-left in scene pixels, so exemplar keypoints detected on the
    crop come back in the same scene coordinate frame as the scene keypoints -- no per-call
    offset arithmetic is needed anywhere downstream.
    """
    keypoints, descriptors = backend.detector.detectAndCompute(gray, None)
    if descriptors is None or len(keypoints) == 0:
        empty_desc: npt.NDArray[np.generic] = np.empty((0, 0), dtype=np.float32)
        return _Keypoints(np.empty((0, 2), np.float64), None, None, empty_desc)
    ox, oy = origin_xy
    xy = np.array([[kp.pt[0] + ox, kp.pt[1] + oy] for kp in keypoints], dtype=np.float64)
    scale = np.array([kp.size for kp in keypoints], dtype=np.float64)
    angle = np.array([kp.angle for kp in keypoints], dtype=np.float64)
    return _Keypoints(xy, scale, angle, np.asarray(descriptors))


# ------------------------------------------------------------------------------ matching


@dataclass(frozen=True)
class _Correspondence:
    """One crop-keypoint to scene-keypoint match, carried through voting and RANSAC.

    ``rank`` is which nearest neighbour this was (0 for the closest). Because the standard ratio
    test is disabled, the rank is the honest replacement for the discarded best/second signal.
    """

    index: int  # position in the correspondence list; how a peak's members are looked up
    crop_xy: tuple[float, float]
    scene_xy: tuple[float, float]
    crop_scale: float | None
    crop_angle: float | None
    scene_scale: float | None
    scene_angle: float | None
    distance: float
    rank: int


_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.int64)


def _pairwise_distances(
    query: npt.NDArray[np.generic],
    train: npt.NDArray[np.generic],
    metric: Literal["l2", "hamming"],
) -> npt.NDArray[np.float64]:
    """Full ``(n_query, n_train)`` descriptor distance matrix for the backend's metric.

    L2 is computed via the ``|a|^2 + |b|^2 - 2ab`` expansion (clamped at 0 before the sqrt to
    swallow floating-point dust); Hamming XORs the uint8 descriptors and counts bits with a
    256-entry popcount table. Correlating over the FULL descriptor set is deliberate -- these
    are the many-to-many distances the top-k step ranks, and cropping them would silently change
    which neighbours win.
    """
    if metric == "l2":
        q = query.astype(np.float64)
        t = train.astype(np.float64)
        q_sq = np.sum(q * q, axis=1, keepdims=True)
        t_sq = np.sum(t * t, axis=1, keepdims=True).T
        gram = q @ t.T
        sq = np.maximum(q_sq + t_sq - 2.0 * gram, 0.0)
        l2: npt.NDArray[np.float64] = np.sqrt(sq)
        return l2
    # hamming: XOR every pair of uint8 rows, look bit-counts up in the popcount table, sum.
    q_bytes = query.astype(np.uint8)
    t_bytes = train.astype(np.uint8)
    xor = q_bytes[:, None, :] ^ t_bytes[None, :, :]
    hamming: npt.NDArray[np.float64] = _POPCOUNT[xor].sum(axis=2).astype(np.float64)
    return hamming


def _topk_neighbours(
    distances: npt.NDArray[np.float64], k_plus_1: int
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """Return the indices and distances of the ``k_plus_1`` nearest train rows per query row.

    The (k+1)-th neighbour is fetched alongside the top-k so the optional k+1 ratio and the
    ``k_ceiling_hit`` metric can be computed without a second pass. Ties are broken by train
    index (a stable ``argsort`` after ``argpartition``) so the output is deterministic.
    """
    n_train = distances.shape[1]
    take = min(k_plus_1, n_train)
    part = np.argpartition(distances, take - 1, axis=1)[:, :take]
    rows = np.arange(distances.shape[0])[:, None]
    part_dist = distances[rows, part]
    order = np.argsort(part_dist, axis=1, kind="stable")
    idx = part[rows, order]
    dist = part_dist[rows, order]
    return idx.astype(np.int64), dist


@dataclass(frozen=True)
class _MatchResult:
    """Correspondences plus the diagnostics the matching step is responsible for."""

    correspondences: tuple[_Correspondence, ...]
    k_ceiling_hit: int  # crop keypoints whose k-th neighbour is still strong (truncated repeats)
    n_dropped_kplus1: int  # crop keypoints dropped by the optional k+1 ratio
    n_crop_matched: int  # crop keypoints that contributed at least one correspondence


def _match_top_k(
    crop: _Keypoints,
    scene: _Keypoints,
    metric: Literal["l2", "hamming"],
    k: int,
    use_kplus1_ratio: bool,
    kplus1_ratio: float,
) -> _MatchResult:
    """Many-to-many top-k matching with the standard Lowe ratio test DISABLED.

    For each crop keypoint we take its top-``k`` scene neighbours **unconditionally** -- there is
    no best/second ratio anywhere in this function, by design (repeated instances are exactly the
    multi-good-match signature the standard ratio test suppresses). The only optional test is the
    **k+1 ratio**: when enabled, a crop keypoint is dropped if its k-th neighbour is within
    ``kplus1_ratio`` of its (k+1)-th, i.e. the k cutoff is not separating real matches from the
    tail. Independently of that option, every crop keypoint whose k-th neighbour is still strong
    is counted towards ``k_ceiling_hit`` so a truncated instance set is visible in diagnostics.
    """
    if crop.count == 0 or scene.count == 0:
        return _MatchResult((), 0, 0, 0)

    idx, dist = _topk_neighbours(
        _pairwise_distances(crop.descriptors, scene.descriptors, metric), k + 1
    )
    available = idx.shape[1]

    correspondences: list[_Correspondence] = []
    k_ceiling_hit = 0
    n_dropped = 0
    n_matched = 0
    for i in range(crop.count):
        row_idx = idx[i]
        row_dist = dist[i]
        n_take = min(k, available)

        # The k-vs-(k+1) comparison: only meaningful when a (k+1)-th neighbour exists.
        ceiling_strong = False
        if available > k:
            d_k = float(row_dist[k - 1])
            d_kp1 = float(row_dist[k])
            ceiling_strong = d_k >= _K_CEILING_RATIO * d_kp1 if d_kp1 > 0.0 else d_k == 0.0
            if ceiling_strong:
                k_ceiling_hit += 1
            if use_kplus1_ratio and (d_k >= kplus1_ratio * d_kp1 if d_kp1 > 0.0 else d_k == 0.0):
                n_dropped += 1
                continue  # drop this crop keypoint's correspondences entirely

        n_matched += 1
        for rank in range(n_take):
            j = int(row_idx[rank])
            correspondences.append(
                _Correspondence(
                    index=len(correspondences),
                    crop_xy=(float(crop.xy[i, 0]), float(crop.xy[i, 1])),
                    scene_xy=(float(scene.xy[j, 0]), float(scene.xy[j, 1])),
                    crop_scale=None if crop.scale is None else float(crop.scale[i]),
                    crop_angle=None if crop.angle is None else float(crop.angle[i]),
                    scene_scale=None if scene.scale is None else float(scene.scale[j]),
                    scene_angle=None if scene.angle is None else float(scene.angle[j]),
                    distance=float(row_dist[rank]),
                    rank=rank,
                )
            )

    logger.debug(
        "sparse-geo matching: {} correspondences from {} crop keypoints "
        "(k_ceiling_hit={}, dropped_kplus1={})",
        len(correspondences),
        crop.count,
        k_ceiling_hit,
        n_dropped,
    )
    return _MatchResult(tuple(correspondences), k_ceiling_hit, n_dropped, n_matched)


# ---------------------------------------------------------------- generalized Hough voting


@dataclass(frozen=True)
class _Vote:
    """One correspondence's (or pair's) prediction of an instance's pose in the scene.

    ``px, py`` is the predicted **object-centre** location in scene pixels; ``log_scale`` and
    ``theta_deg`` are the hypothesized scale (natural log) and rotation. ``members`` are the
    indices of the correspondences that produced this vote -- one for the single/translation
    modes, two for a pairwise vote -- so the winning peak knows exactly which correspondences to
    hand to RANSAC.
    """

    px: float
    py: float
    log_scale: float
    theta_deg: float
    members: tuple[int, ...]


def _proper_similarity_2pt(
    p1: tuple[float, float],
    p2: tuple[float, float],
    q1: tuple[float, float],
    q2: tuple[float, float],
) -> tuple[complex, complex] | None:
    """Solve the proper (orientation-preserving) 4-DoF similarity ``q = a*p + b`` from two pairs.

    Writing points as complex numbers turns a 4-DoF similarity into a single complex-linear map:
    ``a = s*e^{i*theta}`` carries scale and rotation, ``b`` the translation. Two point pairs
    determine it exactly: ``a = (q2 - q1) / (p2 - p1)``, ``b = q1 - a*p1``. Returns ``None`` when
    the two source points coincide (an undetermined transform). This is the *proper* branch only;
    the reflection branch lives in the RANSAC layer, where the mirror check needs it.
    """
    dp = complex(p2[0] - p1[0], p2[1] - p1[1])
    if abs(dp) < 1e-12:
        return None
    dq = complex(q2[0] - q1[0], q2[1] - q1[1])
    a = dq / dp
    b = complex(q1[0], q1[1]) - a * complex(p1[0], p1[1])
    return a, b


def _vote_single_4dof(corr: _Correspondence, centre: tuple[float, float]) -> _Vote | None:
    """Single-correspondence vote using the keypoint frame (Lowe's original, free).

    A framed keypoint carries ``(x, y, scale, orientation)``, so ONE correspondence determines a
    full similarity: relative scale ``s = scene_scale / crop_scale`` and relative rotation
    ``theta = scene_angle - crop_angle``. The offset from the crop keypoint to the object centre,
    transformed by that similarity, predicts the centre in the scene. Requires a frame; the caller
    guarantees it (and raises otherwise) before this is reached.
    """
    if corr.crop_scale is None or corr.crop_angle is None:
        raise ValueError("single-4dof voting requires framed keypoints (scale + orientation)")
    if corr.scene_scale is None or corr.scene_angle is None:
        raise ValueError("single-4dof voting requires framed keypoints (scale + orientation)")
    if corr.crop_scale <= 0.0 or corr.scene_scale <= 0.0:
        return None
    s = corr.scene_scale / corr.crop_scale
    theta = corr.scene_angle - corr.crop_angle
    rad = np.radians(theta)
    cos_t, sin_t = float(np.cos(rad)), float(np.sin(rad))
    dxl = centre[0] - corr.crop_xy[0]
    dyl = centre[1] - corr.crop_xy[1]
    px = corr.scene_xy[0] + s * (cos_t * dxl - sin_t * dyl)
    py = corr.scene_xy[1] + s * (sin_t * dxl + cos_t * dyl)
    return _Vote(px, py, float(np.log(s)), theta, (corr.index,))


def _vote_translation_2dof(corr: _Correspondence, centre: tuple[float, float]) -> _Vote:
    """Translation-only vote (any backend), assuming instances share the exemplar scale/rotation.

    The predicted centre is the exemplar centre plus the crop->scene translation of this
    correspondence. Scale and rotation are pinned at the identity (``log_scale = 0``,
    ``theta = 0``), which is correct and fast for the near-identical case.
    """
    tx = corr.scene_xy[0] - corr.crop_xy[0]
    ty = corr.scene_xy[1] - corr.crop_xy[1]
    return _Vote(centre[0] + tx, centre[1] + ty, 0.0, 0.0, (corr.index,))


@dataclass(frozen=True)
class _VoteCast:
    """The votes plus the pairwise-sampling diagnostics the voting step is responsible for."""

    votes: tuple[_Vote, ...]
    pairwise_pairs_sampled: int  # 0 unless voting_mode == pairwise-4dof
    pairwise_capped: bool  # True when the O(n^2) pair set was capped by config


def _cast_votes(
    mode: Literal["single-4dof", "translation-2dof", "pairwise-4dof"],
    correspondences: tuple[_Correspondence, ...],
    centre: tuple[float, float],
    *,
    has_frame: bool,
    pairwise_cap: int,
    rng: np.random.Generator,
) -> _VoteCast:
    """Turn correspondences into pose votes under the selected voting mode (METHOD-04a).

    ``single-4dof`` is valid only for a framed backend and **raises** on a frameless one -- a
    config that is accepted and then quietly does something else is worse than a refusal.
    ``translation-2dof`` and ``pairwise-4dof`` work for any backend; ``pairwise-4dof`` samples
    correspondence pairs up to ``pairwise_cap`` (it is O(n^2)) and records the cap so a slow run
    is explained rather than mysterious.
    """
    if mode == "single-4dof":
        if not has_frame:
            raise ValueError(
                "voting_mode='single-4dof' requires a backend whose keypoints carry a frame "
                "(scale + orientation); this backend is frameless. Use 'translation-2dof' or "
                "'pairwise-4dof' instead."
            )
        single = [_vote_single_4dof(corr, centre) for corr in correspondences]
        return _VoteCast(tuple(v for v in single if v is not None), 0, False)

    if mode == "translation-2dof":
        translation = [_vote_translation_2dof(corr, centre) for corr in correspondences]
        return _VoteCast(tuple(translation), 0, False)

    # pairwise-4dof: each pair of correspondences determines a 4-DoF similarity.
    n = len(correspondences)
    total_pairs = n * (n - 1) // 2
    if total_pairs <= pairwise_cap:
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        capped = False
    else:
        sampled: set[tuple[int, int]] = set()
        while len(sampled) < pairwise_cap:
            i = int(rng.integers(0, n))
            j = int(rng.integers(0, n))
            if i != j:
                sampled.add((min(i, j), max(i, j)))
        pairs = sorted(sampled)  # sort for determinism regardless of set iteration order
        capped = True

    votes: list[_Vote] = []
    for i, j in pairs:
        ci, cj = correspondences[i], correspondences[j]
        model = _proper_similarity_2pt(ci.crop_xy, cj.crop_xy, ci.scene_xy, cj.scene_xy)
        if model is None:
            continue
        a, b = model
        if abs(a) <= 0.0:
            continue
        predicted = a * complex(centre[0], centre[1]) + b
        votes.append(
            _Vote(
                predicted.real,
                predicted.imag,
                float(np.log(abs(a))),
                float(np.degrees(np.angle(a))),
                (ci.index, cj.index),
            )
        )
    return _VoteCast(tuple(votes), len(pairs), capped)


def _soft_neighbours(value: float, width: float) -> tuple[tuple[int, float], tuple[int, float]]:
    """Split ``value`` across its two nearest bins (linear soft assignment).

    Returns ``((lower_bin, weight), (upper_bin, weight))`` where the weights sum to 1. Soft
    binning is required (Lowe's boundary fix): a vote near a bin edge would otherwise fall
    entirely into one bin and no peak would clear the floor when instances straddle a boundary.
    """
    coord = value / width
    lower = int(np.floor(coord))
    frac = coord - lower
    return (lower, 1.0 - frac), (lower + 1, frac)


def _accumulate_votes(
    votes: tuple[_Vote, ...], base_location_width: float
) -> tuple[dict[tuple[int, int, int, int], float], dict[tuple[int, int, int, int], list[int]]]:
    """Accumulate votes into a hash-table pose histogram with soft binning and circular theta.

    Bins are Lowe's §7.3 widths: 30 degrees orientation, a factor of 2 in scale (so log-scale is
    binned by ``log 2``), and ``0.25 x max projected crop dimension`` location. The location bin
    width is **scale-dependent** -- ``base_location_width * 2**scale_bin`` -- which is exactly why
    votes live in a **dict keyed by ``(x, y, scale, theta)``**, not a dense array. Each vote is
    soft-assigned into the 2 nearest bins per dimension (16 bins in 4-DoF), and **theta wraps
    circularly** modulo the 12 orientation bins so a vote near 0/360 reaches the adjacent bin.
    """
    log_two = float(np.log(_SCALE_BIN_FACTOR))
    weight: dict[tuple[int, int, int, int], float] = {}
    members: dict[tuple[int, int, int, int], list[int]] = {}

    for vote in votes:
        # theta: circular, modulo _N_THETA_BINS bins of _THETA_BIN_DEG each.
        theta = vote.theta_deg % 360.0
        t_coord = theta / _THETA_BIN_DEG
        t_lower = int(np.floor(t_coord))
        t_frac = t_coord - t_lower
        theta_bins = (
            (t_lower % _N_THETA_BINS, 1.0 - t_frac),
            ((t_lower + 1) % _N_THETA_BINS, t_frac),
        )
        scale_bins = _soft_neighbours(vote.log_scale, log_two)

        for s_idx, s_w in scale_bins:
            # Location bin width grows with the hypothesized scale (Lowe): within one scale bin
            # all votes share this width, so their location bins align and can cluster.
            loc_w = base_location_width * (_SCALE_BIN_FACTOR**s_idx)
            x_bins = _soft_neighbours(vote.px, loc_w)
            y_bins = _soft_neighbours(vote.py, loc_w)
            for t_idx, t_w in theta_bins:
                for x_idx, x_w in x_bins:
                    for y_idx, y_w in y_bins:
                        w = s_w * t_w * x_w * y_w
                        if w <= 0.0:
                            continue
                        key = (x_idx, y_idx, s_idx, t_idx)
                        weight[key] = weight.get(key, 0.0) + w
                        members.setdefault(key, []).extend(vote.members)
    return weight, members


def _neighbourhood(key: tuple[int, int, int, int]) -> list[tuple[int, int, int, int]]:
    """The 3^4 bins adjacent to ``key`` (inclusive), with theta wrapping circularly.

    Used to de-duplicate peaks: adjacent bins describe the same cluster and must not both be
    reported. Location and scale simply step +/-1; theta steps +/-1 modulo the 12 orientation
    bins so the neighbourhood of bin 0 includes bin 11.
    """
    x, y, s, t = key
    out: list[tuple[int, int, int, int]] = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for ds in (-1, 0, 1):
                for dt in (-1, 0, 1):
                    out.append((x + dx, y + dy, s + ds, (t + dt) % _N_THETA_BINS))
    return out


@dataclass(frozen=True)
class _Peak:
    """One hypothesized instance cluster: a pose-space bin plus its member correspondences."""

    votes: float
    dx: float
    dy: float
    log_scale: float
    theta_deg: float
    member_indices: tuple[int, ...]


def _enumerate_peaks(
    weight: dict[tuple[int, int, int, int], float],
    members: dict[tuple[int, int, int, int], list[int]],
    min_votes: int,
    base_location_width: float,
) -> tuple[_Peak, ...]:
    """Enumerate pose-space peaks: bins with >= ``min_votes`` weight, de-duplicated by 3^4.

    Bins are taken strongest-first; a bin is skipped when an already-accepted peak lies in its
    3^4 neighbourhood, so adjacent bins describing one cluster are not reported twice. Each peak
    carries the union of its member correspondence indices (what per-peak RANSAC verifies) and a
    representative pose at the bin's lower corner (diagnostics only).
    """
    candidates = sorted(
        (key for key, w in weight.items() if w >= min_votes),
        key=lambda key: (-weight[key], key),
    )
    accepted: list[_Peak] = []
    accepted_keys: set[tuple[int, int, int, int]] = set()
    log_two = float(np.log(_SCALE_BIN_FACTOR))
    for key in candidates:
        if any(neighbour in accepted_keys for neighbour in _neighbourhood(key)):
            continue
        accepted_keys.add(key)
        x_idx, y_idx, s_idx, t_idx = key
        loc_w = base_location_width * (_SCALE_BIN_FACTOR**s_idx)
        unique_members = tuple(sorted(set(members[key])))
        accepted.append(
            _Peak(
                votes=weight[key],
                dx=x_idx * loc_w,
                dy=y_idx * loc_w,
                log_scale=s_idx * log_two,
                theta_deg=(t_idx * _THETA_BIN_DEG),
                member_indices=unique_members,
            )
        )
    return tuple(accepted)
