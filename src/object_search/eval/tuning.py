"""Per-method domain threshold tuning: tune on val, freeze, report tuned-vs-default on test.

The research sweep (:func:`object_search.eval.benchmark.run_research_sweep`) scores every method at
its **default** config. That answers "how do the methods do out of the box on floor plans", but not
"how good is each method once its acceptance threshold is adapted to this domain" -- which is the
question that actually picks a method to ship. This module answers the second one, honestly:

1. **Tune on val.** For each method, sweep a small explicit grid of acceptance knobs
   (:data:`_TUNING_GRIDS`) on the dataset's ``val`` split and pick the config that maximises
   **F1 @ IoU 0.5** -- the operating-point metric the product cares about (find all the doors
   without junk). The grids are broadened for the floor-plan domain: each method sweeps **two
   knobs** (a primary acceptance knob crossed with a second recall/precision knob), stated per
   method in :data:`_TUNING_GRIDS`. Nothing here ever reads ``test``.
2. **Freeze + evaluate on test.** Run the frozen (tuned) config on ``test`` once. Also run the
   method's **default** config on ``test``. Reporting both side by side is the point: it shows which
   method wins on floor plans *and* how much domain tuning each one needed to get there (a method
   that barely moves is robust; one that jumps was mis-calibrated for this domain).

Why hand-written grids per method? Each method gates acceptance differently -- a calibrated score
floor (``ncc``/``mosse``/``dino-dense``/``owlv2`` ``retain_frac``), a geometric inlier count
(``sparse-geo`` ``min_inliers``), a retrieval cosine floor (``propose-retrieve``
``similarity_floor``) -- and the knob(s) that trade recall against precision are method-specific.
The floor-plan grids pair each method's primary knob with a second one (a symbol-matched
``scales`` bank and ``nms_iou`` for the correlation methods, ``nms_iou`` for the geometric/retrieval
methods, ``max_box_area_frac`` + ``query_iou_frac`` for OWLv2), each entry a multi-key override dict
validated through the method's own frozen ``config_model``. A hand-written grid keeps this readable
and editable (a practitioner tunes the numbers here), rather than hiding the search behind a generic
optimiser. The tuned config is always an instance of the method's own frozen ``config_model``, so
tuning never touches a method file -- it only feeds a different config through the additive
``config`` param on :func:`object_search.eval.benchmark.run_research_benchmark`.

Reproducibility: the grids are fixed, the exemplar sampler is seeded (D-11), and val/test come from
the committed split manifest, so the same dataset bytes + seed reproduce the same frozen configs and
the same tuned-vs-default table byte for byte.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel

from object_search.eval.benchmark import run_research_benchmark
from object_search.eval.sampling import ExemplarSelection
from object_search.provenance import current_git_sha, repo_root
from object_search.search import get_method

# The six methods, tuned in registry order. Kept here (not imported from benchmark) so the tuning
# run set is edited in one obvious place. Public because the CLI's ``--methods`` option needs the
# same default to fall back to, and a private cross-module import would be worse than exporting it.
DEFAULT_TUNING_METHODS: tuple[str, ...] = (
    "ncc",
    "mosse",
    "sparse-geo",
    "dino-dense",
    "propose-retrieve",
    "owlv2-oneshot",
)

# Per-method DOMAIN TUNING GRID: a small, explicit grid of config overrides swept on val, argmax
# F1 @ IoU 0.5. Each entry is a dict applied atop the method's defaults (validated through the
# method's own frozen config_model), so a multi-key entry is just a config with several fields set.
# The grids are broadened for the floor-plan domain: each method now sweeps TWO knobs -- a primary
# acceptance knob crossed with a second knob that trades recall against precision a different way --
# stated method by method below. Edit the tuples here to widen or refine the search.
#
# ncc / mosse -- three knobs, because a floor-plan symbol is small and near-fixed-scale:
#   * scales: SYMBOL-MATCHED banks, biased tighter/finer than the method default (0.75-1.3, too
#     wide for a fixed-scale plan and a source of rotated-template false peaks). (1.0,) is the pure
#     fixed-scale bank; the fine +/-10% triple tolerates slight symbol-size drift without the wide
#     default's spurious peaks.
#   * retain_frac: self-similarity acceptance fraction; lower keeps more of the transformed tail
#     (recall up, precision down). Widened around the method default.
#   * nms_iou: overlap ceiling for accepted boxes; tight (0.3) splits touching symbols, loose (0.5)
#     merges them -- the recall/precision trade at the localisation level.
_NCC_SCALE_SETS: tuple[tuple[float, ...], ...] = ((1.0,), (0.9, 1.0, 1.1))
_NCC_MOSSE_RETAIN: tuple[float, ...] = (0.25, 0.35, 0.45, 0.55, 0.65)
_NCC_MOSSE_NMS: tuple[float, ...] = (0.3, 0.5)


def _correlation_grid() -> tuple[dict[str, object], ...]:
    """Build a FRESH scales x retain_frac x nms_iou grid for one correlation method.

    ``ncc`` and ``mosse`` start from the same three knobs but must NOT share one grid object.
    Both config models are ``extra="forbid"``, so the moment either method grows a method-only
    knob (an ``ncc`` rotation-bank variant, a ``mosse`` filter knob) a shared grid would feed
    that key into the *other* method's validator and raise. Each call returns independent
    tuples and dicts, so the two grids below can diverge freely.
    """
    return tuple(
        {"scales": scales, "retain_frac": retain, "nms_iou": nms}
        for scales in _NCC_SCALE_SETS
        for retain in _NCC_MOSSE_RETAIN
        for nms in _NCC_MOSSE_NMS
    )


# ``ncc``-only ADDITIVE block from the floor-plan domain investigation (quick task 260730-vx4,
# see EXPERIMENTS.md in that quick task's directory). A floor-plan door/window symbol sits on
# whichever wall it is drawn on, so an instance on a perpendicular wall can be ~90 deg off the
# exemplar -- outside the shipped +/-35 deg bank (`NCCConfig.angles_deg`'s default). A pure
# 4-angle CARDINAL bank (0/90/180/270) measured a clear win over both the shipped bank and wider
# continuous banks (a 28-angle cardinal-x-fine sub-bank, a uniform 30 deg spacing) on this domain:
# floorplans-door test F1 0.164 -> 0.355, floorplans-window 0.222 -> 0.272 -- floor-plan walls are
# discretely orthogonal, not continuously rotated, so a small precise cardinal set beats a dense
# sweep. `mirror` (a horizontally-flipped template sibling, for domains with bilateral symmetry
# like door swing direction) is included as its own axis rather than a fixed choice because its
# effect measured genuinely mixed: a statistical tie for doors (test F1 0.357 vs 0.355, trading
# precision for recall) and a net-negative for windows (val F1 0.276 vs 0.299) -- argmax-on-val
# lets each dataset pick honestly instead of hand-picking one global default. Fixed at a single
# scale (the floor-plan symbols are near-fixed-scale, matched by the existing (1.0,) scale option)
# and the tighter nms_iou (0.3), since the investigation swept those two independently of scale/nms
# and widening this block's own scale x nms cross would balloon the grid for no measured benefit.
_NCC_CARDINAL_BANK: tuple[float, ...] = (0.0, 90.0, 180.0, 270.0)
_NCC_MIRROR_OPTIONS: tuple[bool, ...] = (False, True)


def _ncc_grid() -> tuple[dict[str, object], ...]:
    """``ncc``'s grid: the original scales x retain_frac x nms_iou sweep, PLUS an additive
    cardinal-rotation-bank x mirror block (see the module comment above `_NCC_CARDINAL_BANK`).

    Additive, not a replacement: the shipped angle bank (mirror off) stays available as before, so
    this can only match or beat the pre-existing grid on any dataset, never regress it.
    """
    return _correlation_grid() + tuple(
        {
            "scales": (1.0,),
            "retain_frac": retain,
            "nms_iou": 0.3,
            "angles_deg": _NCC_CARDINAL_BANK,
            "mirror": mirror,
        }
        for retain in _NCC_MOSSE_RETAIN
        for mirror in _NCC_MIRROR_OPTIONS
    )


# ``mosse``-only ADDITIVE block from the floor-plan domain investigation (quick task 260730-w9s,
# see EXPERIMENTS.md in that quick task's directory). Mirrors the sibling `ncc` investigation's
# finding: a pure 4-angle CARDINAL bank (0/90/180/270 deg), with `n_angle_groups` scaled to match
# (4 groups -- one per cardinal, so each sub-filter stays sharp), beat both the shipped +/-35 deg
# bank and wider continuous banks on BOTH floor-plan classes -- floorplans-door test F1 0.201 ->
# 0.414, floorplans-window 0.077 -> 0.141. Widening the bank WITHOUT scaling groups proportionally
# (the naive "28 angles, still 4 groups" trial) reproduces the already-measured-bad one-blurry-
# filter failure mode and must not be mistaken for a fair test -- see EXPERIMENTS.md E1.
# `mirror` (a horizontally-flipped verify-side re-score template, for domains with bilateral
# symmetry like door swing direction) is included as its own axis: unlike `ncc`'s near-tie, it is a
# STRONG additional win for doors (F1 0.414 -> 0.509 val-argmax) and a mild net-negative for
# windows (mirror stays off there) -- argmax-on-val lets each dataset pick honestly. Fixed at a
# single scale and the tighter nms_iou, matching the investigation's own sweep (which held those
# fixed while varying the bank/groups/mirror axes).
_MOSSE_CARDINAL_BANK: tuple[float, ...] = (0.0, 90.0, 180.0, 270.0)
_MOSSE_CARDINAL_GROUPS = 4
_MOSSE_MIRROR_OPTIONS: tuple[bool, ...] = (False, True)


def _mosse_grid() -> tuple[dict[str, object], ...]:
    """``mosse``'s grid: the original scales x retain_frac x nms_iou sweep, PLUS an additive
    cardinal-rotation-bank (with matched `n_angle_groups`) x mirror block (see the module comment
    above `_MOSSE_CARDINAL_BANK`).

    Additive, not a replacement: the shipped bank/groups (mirror off) stay available as before, so
    this can only match or beat the pre-existing grid on any dataset, never regress it.
    """
    return _correlation_grid() + tuple(
        {
            "scales": (1.0,),
            "retain_frac": retain,
            "nms_iou": 0.3,
            "train_angles_deg": _MOSSE_CARDINAL_BANK,
            "n_angle_groups": _MOSSE_CARDINAL_GROUPS,
            "mirror": mirror,
        }
        for retain in _NCC_MOSSE_RETAIN
        for mirror in _MOSSE_MIRROR_OPTIONS
    )


# ``propose-retrieve``-only ADDITIVE block from the floor-plan domain investigation (quick task
# 260812-m8m, see EXPERIMENTS.md in that quick task's directory and
# docs/reports/propose-retrieve-floorplans-improvement.md). FastSAM's everything-mode proposal COUNT
# scales with image AREA (r = +0.59 over 84 plans), not with instance count (r = +0.22), so a
# crowded CAD plan gets ~46 proposals for ~15 doors and the PROPOSAL stage caps recall at 0.405
# before retrieval ever runs. `proposal_conf` (FastSAM's objectness gate, default 0.4) is the
# lever that lifts that cap, and it had never been tuned for this domain: lowering it to 0.10 takes
# proposal-stage recall 0.498 -> 0.821 and floorplans-door test F1 0.481 -> 0.597 (+24% relative),
# almost all of it recall (0.399 -> 0.674) for a precision cost of 0.604 -> 0.536.
#
# Swept jointly with `similarity_floor` rather than alone, because opening the gate changes the
# proposal distribution the floor has to reject false positives from (46.5 -> 161.2 proposals/plan),
# so the floor's optimum genuinely might move. Measured: it does not -- the argmax floor is 0.70,
# the SHIPPED default, on both the old and the new distribution. F1 is monotone on both axes
# (falling in floor at every conf, and 0.10 > 0.20 > 0.30 at every floor), so the argmax is a
# corner of the swept region, not an unstable ridge -- unlike the owlv2 doors experience in
# docs/reports/owlv2-floorplans-improvement.md. `floor 0.85` is omitted (measured worst by a wide
# margin, val F1 0.247) and `nms_iou` is fixed at the tighter 0.3, which won every pair in B1's
# 6x2 sweep; both were held fixed in the investigation's own grid rather than re-crossed here.
#
# Additive, not a replacement: the pre-existing similarity_floor x nms_iou entries stay untouched
# and available (they include the shipped default at conf 0.4), so this can only match or beat the
# previous grid on any dataset, never regress it. It ships as a GRID entry and NOT as a changed
# `ProposeRetrieveConfig.proposal_conf` default deliberately -- a lower gate trades textured
# precision for recall, which the general-case pass already measured and declined
# (docs/reports/propose-retrieve-improvement.md's closing note), so this stays a domain lever that a
# floor-plan tuning run opts into, leaving the chipset/textured/synthetic regimes byte-identical.
#
# NOT part of this block, and deliberately so: the SAHI-style tiling fields
# (`proposal_tiling` et al.) built in this same quick task. At a MATCHED proposal budget the
# objectness gate beat tiling by +0.233 mean proposal recall at a THIRD of the latency, so no tiled
# cell would ever win this grid. See the report for the full measured-and-rejected record.
_PROPOSE_RETRIEVE_FLOOR: tuple[float, ...] = (0.4, 0.5, 0.6, 0.7, 0.8, 0.85)
_PROPOSE_RETRIEVE_NMS: tuple[float, ...] = (0.3, 0.5)
_PROPOSE_RETRIEVE_CONF: tuple[float, ...] = (0.1, 0.2, 0.3)
_PROPOSE_RETRIEVE_DOMAIN_FLOOR: tuple[float, ...] = (0.7, 0.75, 0.8)


def _propose_retrieve_grid() -> tuple[dict[str, object], ...]:
    """``propose-retrieve``'s grid: the original ``similarity_floor`` x ``nms_iou`` sweep, PLUS an
    additive ``proposal_conf`` x ``similarity_floor`` block (see the module comment above
    ``_PROPOSE_RETRIEVE_FLOOR``).

    Additive, not a replacement: the shipped ``proposal_conf`` (0.4) stays available through the
    original block, so this can only match or beat the pre-existing grid, never regress it. Every
    one of the nine additive cells traces to a committed val run in quick task 260812-m8m.
    """
    # Both blocks are annotated ``dict[str, object]`` rather than inferred: every value here is a
    # float, so mypy would infer the invariant ``dict[str, float]`` and reject the wider return
    # type that `_TUNING_GRIDS` and `tune_method`'s ``grid=`` parameter share with every other grid.
    base: tuple[dict[str, object], ...] = tuple(
        {"similarity_floor": floor, "nms_iou": nms}
        for floor in _PROPOSE_RETRIEVE_FLOOR
        for nms in _PROPOSE_RETRIEVE_NMS
    )
    domain: tuple[dict[str, object], ...] = tuple(
        {"proposal_conf": conf, "similarity_floor": floor, "nms_iou": 0.3}
        for conf in _PROPOSE_RETRIEVE_CONF
        for floor in _PROPOSE_RETRIEVE_DOMAIN_FLOOR
    )
    return base + domain


_NCC_GRID: tuple[dict[str, object], ...] = _ncc_grid()
_MOSSE_GRID: tuple[dict[str, object], ...] = _mosse_grid()
_PROPOSE_RETRIEVE_GRID: tuple[dict[str, object], ...] = _propose_retrieve_grid()

_TUNING_GRIDS: Mapping[str, tuple[dict[str, object], ...]] = {
    "ncc": _NCC_GRID,
    "mosse": _MOSSE_GRID,
    # sparse-geo -- min_inliers (RANSAC inliers to accept; higher -> stricter, the primary knob,
    # widened down to 2 and up to 10) crossed with nms_iou (duplicate-instance suppression).
    "sparse-geo": tuple(
        {"min_inliers": inliers, "nms_iou": nms}
        for inliers in (2, 3, 4, 5, 6, 8, 10)
        for nms in (0.3, 0.5)
    ),
    # dino-dense -- retain_frac: DINO dense-feature score floor (contrast-calibrated). Lower ->
    # stricter. Single knob (the letterbox fix is Task 4; dense-feature acceptance is the trade).
    "dino-dense": tuple({"retain_frac": v} for v in (0.5, 0.6, 0.7, 0.8, 0.9)),
    # propose-retrieve -- similarity_floor (min cosine to a retrieved proposal embedding; higher ->
    # stricter, widened from 0.4) crossed with nms_iou (collapses FastSAM over-segmentation), PLUS
    # an additive proposal_conf x similarity_floor block for the floor-plan domain, where the
    # PROPOSAL stage (not retrieval) is what caps recall -- see above `_PROPOSE_RETRIEVE_FLOOR`.
    "propose-retrieve": _PROPOSE_RETRIEVE_GRID,
    # owlv2-oneshot -- max_box_area_frac (drop boxes bigger than this fraction of the image; the
    # whole-frame-box filter) crossed with query_iou_frac (how wide the query-patch set is), the two
    # knobs that most move floor-plan precision/recall. retain_frac stays at the method default.
    # The grid was widened DOWN from {0.1, 0.25, 0.5} after a debug-image inspection (see
    # docs/reports/owlv2-floorplans-improvement.md) showed the residual floor-plan false positives
    # are large room/wall-sized rectangles, not small symbol-sized boxes -- CAD-symbol scale is a
    # few percent of the plan at most (docs/eval/floorplans-findings.md's dataset statistics), so
    # 0.1 (10%) was still far too generous a cap for this domain and the old grid never tried lower.
    "owlv2-oneshot": tuple(
        {"max_box_area_frac": area, "query_iou_frac": query}
        for area in (0.005, 0.007, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5)
        for query in (0.6, 0.8, 0.9)
    ),
}


def _f1_sort_key(overall: Mapping[str, Any]) -> float:
    """F1 as a sortable float: the abstention ``None`` (nothing scored) sorts below any real F1.

    The tuning objective is F1 @ IoU 0.5; a config that returns nothing pools to F1 ``None`` (not
    ``0``), which must never win a tie against a config that actually found instances.
    """
    f1 = overall.get("f1")
    return float(f1) if isinstance(f1, int | float) else -1.0


def _evaluate(
    method: str,
    dataset: str,
    split: str,
    split_root: Path,
    *,
    config: BaseModel | None,
    exemplar_count: int,
    iou_threshold: float,
    seed: int,
    manifest_root: Path | None,
    exemplar_selection: ExemplarSelection = "seeded-random",
) -> dict[str, Any]:
    """Run one (method, config) over a split and return its pooled ``overall`` metric block."""
    block = run_research_benchmark(
        method,
        dataset,
        split,
        split_root,
        exemplar_count=exemplar_count,
        iou_threshold=iou_threshold,
        seed=seed,
        manifest_root=manifest_root,
        config=config,
        exemplar_selection=exemplar_selection,
    )
    overall = block["overall"]
    assert isinstance(overall, dict)  # noqa: S101  -- run_research_benchmark always returns it
    return overall


def tune_method(
    method: str,
    dataset: str,
    val_root: Path,
    *,
    exemplar_count: int = 1,
    iou_threshold: float = 0.5,
    seed: int = 0,
    manifest_root: Path | None = None,
    grid: Sequence[dict[str, object]] | None = None,
    exemplar_selection: ExemplarSelection = "seeded-random",
) -> dict[str, Any]:
    """Sweep ``method``'s acceptance knob on ``val``; return the argmax-F1 config plus all trials.

    Args:
        method: Registry key, e.g. ``"ncc"``.
        dataset: Dataset key, e.g. ``"floorplans-door"``.
        val_root: Directory of converted val sidecars + scenes (``datasets/<dataset>/val``).
        exemplar_count: Exemplars per query (1 = the product operating point).
        iou_threshold: IoU for a TP; the tuning metric is F1 at this IoU.
        seed: Config seed for the exemplar sampler (D-11).
        manifest_root: Optional base dir for the committed split manifest (tests use ``tmp_path``).
        grid: Override the built-in grid (tests pass a tiny one); defaults to
            :data:`_TUNING_GRIDS` for ``method``.

    Returns:
        ``{"method", "trials": [{overrides, f1, precision, recall}], "best": {overrides, f1,
        val_overall} | None}``. ``best`` is ``None`` only when the grid is empty.
    """
    spec = get_method(method)
    candidates = tuple(grid) if grid is not None else _TUNING_GRIDS.get(method, ())

    trials: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_key = float("-inf")
    for overrides in candidates:
        config = spec.config_model(**overrides)
        overall = _evaluate(
            method,
            dataset,
            "val",
            val_root,
            config=config,
            exemplar_count=exemplar_count,
            iou_threshold=iou_threshold,
            seed=seed,
            manifest_root=manifest_root,
            exemplar_selection=exemplar_selection,
        )
        trials.append(
            {
                "overrides": dict(overrides),
                "f1": overall.get("f1"),
                "precision": overall.get("precision"),
                "recall": overall.get("recall"),
            }
        )
        key = _f1_sort_key(overall)
        if key > best_key:
            best_key = key
            best = {"overrides": dict(overrides), "f1": overall.get("f1"), "val_overall": overall}

    logger.info(
        "tuning[{}/{}]: best {} (val F1={})",
        method,
        dataset,
        best["overrides"] if best else "<no grid>",
        best["f1"] if best else None,
    )
    return {"method": method, "trials": trials, "best": best}


def _delta(tuned: float | None, default: float | None) -> float | None:
    """Signed change from default to tuned, or ``None`` if either side is unscored."""
    if isinstance(tuned, int | float) and isinstance(default, int | float):
        return float(tuned) - float(default)
    return None


def _tune_methods_at_count(
    dataset: str,
    val_root: Path,
    eval_root: Path,
    *,
    methods: Sequence[str],
    exemplar_count: int,
    iou_threshold: float,
    seed: int,
    eval_split: str,
    manifest_root: Path | None,
    exemplar_selection: ExemplarSelection,
    grids: Mapping[str, Sequence[dict[str, object]]] | None = None,
) -> list[dict[str, Any]]:
    """The per-method tune-on-val, freeze, report-tuned-vs-default-on-test loop at ONE count.

    Factored out so :func:`run_domain_tuning` can call it once (the byte-identical single-count
    report) or once per requested count (the additive nested multi-count report) without
    duplicating the loop.

    ``grids``, when given, overrides :data:`_TUNING_GRIDS` for the named method(s) only (a method
    absent from ``grids`` still uses its built-in grid), threaded straight to :func:`tune_method`'s
    own ``grid=`` parameter. ``None`` (every call site before this parameter existed) reproduces
    today's exact behavior byte-for-byte. It exists so a research script can sweep a method-specific
    variant (e.g. a floor-plan rotation-bank sweep, or a fine-tune's crop-margin fraction) without
    shelling out to the full ``tune-floorplans`` CLI, which always tunes every registered method.
    """
    per_method: list[dict[str, Any]] = []
    for method in methods:
        spec = get_method(method)
        method_grid = grids.get(method) if grids is not None else None
        tuned = tune_method(
            method,
            dataset,
            val_root,
            exemplar_count=exemplar_count,
            iou_threshold=iou_threshold,
            seed=seed,
            manifest_root=manifest_root,
            grid=method_grid,
            exemplar_selection=exemplar_selection,
        )
        best = tuned["best"]
        tuned_config = spec.config_model(**best["overrides"]) if best else None

        tuned_test = _evaluate(
            method,
            dataset,
            eval_split,
            eval_root,
            config=tuned_config,
            exemplar_count=exemplar_count,
            iou_threshold=iou_threshold,
            seed=seed,
            manifest_root=manifest_root,
            exemplar_selection=exemplar_selection,
        )
        default_test = _evaluate(
            method,
            dataset,
            eval_split,
            eval_root,
            config=None,
            exemplar_count=exemplar_count,
            iou_threshold=iou_threshold,
            seed=seed,
            manifest_root=manifest_root,
            exemplar_selection=exemplar_selection,
        )
        delta = _delta(tuned_test.get("f1"), default_test.get("f1"))
        per_method.append(
            {
                "method": method,
                "tuned_overrides": best["overrides"] if best else {},
                "val_f1": best["f1"] if best else None,
                "tuned_test": tuned_test,
                "default_test": default_test,
                "delta_f1": delta,
                "trials": tuned["trials"],
            }
        )
        logger.info(
            "tuning[{}/{}]: test F1 tuned={} default={} (delta={})",
            method,
            dataset,
            tuned_test.get("f1"),
            default_test.get("f1"),
            delta,
        )
    return per_method


def run_domain_tuning(
    dataset: str,
    research_root: Path | str,
    *,
    methods: Sequence[str] = DEFAULT_TUNING_METHODS,
    exemplar_count: int = 1,
    iou_threshold: float = 0.5,
    seed: int = 0,
    tune_split: str = "val",
    eval_split: str = "test",
    manifest_root: Path | None = None,
    exemplar_selection: ExemplarSelection = "seeded-random",
    exemplar_counts: Sequence[int] | None = None,
    grids: Mapping[str, Sequence[dict[str, object]]] | None = None,
    out: str | None = "docs/benchmark/floorplans-tuning-results.json",
) -> dict[str, Any]:
    """Tune every method on ``tune_split`` and report tuned-vs-default on ``eval_split``.

    For each method: pick the argmax-F1 config on val (:func:`tune_method`), freeze it, then score
    both the frozen config and the method's defaults on test. The returned report carries, per
    method, the tuned overrides, the val F1 that selected them, the full tuned and default test
    metric blocks, and the F1 delta -- the tuned-vs-default table.

    Exemplar-count operating points:

    * ``exemplar_counts is None`` (default) -- a single count (``exemplar_count``) is tuned and the
      report has the committed flat shape (top-level ``exemplar_count`` + ``methods``).
      Byte-for-byte unchanged from before this option existed.
    * ``exemplar_counts`` given (e.g. ``(1, 3)``) -- each count is tuned independently and the
      report nests one ``{"exemplar_count", "methods"}`` block per count under ``per_count`` (with a
      top-level ``exemplar_counts`` list), so a 1-vs-3 comparison is one report.

    Args:
        dataset: Dataset key, e.g. ``"floorplans-door"``.
        research_root: Base dir holding ``<dataset>/<split>/`` converted trees (``datasets/``).
        methods: Methods to tune (defaults to all six).
        exemplar_count: Exemplars per query when ``exemplar_counts`` is ``None`` (the flat report).
        iou_threshold: IoU for a TP; the tuning metric is F1 at this IoU.
        seed: Config seed for the exemplar sampler (D-11).
        tune_split / eval_split: Split names; tuning never reads ``eval_split``.
        manifest_root: Optional base dir for the committed split manifests (tests use ``tmp_path``).
        exemplar_selection: Exemplar-ordering mode threaded to the sampler (``"seeded-random"``
            default preserves the committed draw; ``"size-representative"`` seeds from the
            median-area box).
        exemplar_counts: Optional sequence of counts to nest per-count blocks for; ``None`` keeps
            the flat single-count report.
        grids: Optional per-method grid override, e.g. ``{"ncc": ({"angles_deg": (...)}, ...)}``,
            keyed by method name. A method present here is tuned over the supplied grid INSTEAD of
            its :data:`_TUNING_GRIDS` entry; a method absent from the mapping (and the ``None``
            default) keeps its committed grid, so omitting this argument reproduces the previous
            report byte for byte. This is the seam an offline experiment script uses to sweep a
            candidate knob (e.g. a floor-plan rotation-bank sweep, or a fine-tune's crop-margin
            fraction) without editing the committed grids or forking the tuning loop.
        out: Where to write the JSON report (resolved against the repo root when relative). ``None``
            skips the write and only returns the report.

    Returns:
        The report mapping (also written to ``out`` unless ``out`` is ``None``).
    """
    base = Path(research_root)
    if not base.is_absolute():
        base = repo_root() / base
    val_root = base / dataset / tune_split
    eval_root = base / dataset / eval_split

    def _methods_at(count: int) -> list[dict[str, Any]]:
        return _tune_methods_at_count(
            dataset,
            val_root,
            eval_root,
            methods=methods,
            exemplar_count=count,
            iou_threshold=iou_threshold,
            seed=seed,
            eval_split=eval_split,
            manifest_root=manifest_root,
            exemplar_selection=exemplar_selection,
            grids=grids,
        )

    report: dict[str, Any] = {
        "git_sha": current_git_sha(),
        "dataset": dataset,
        "tune_split": tune_split,
        "eval_split": eval_split,
        "iou_threshold": iou_threshold,
        "seed": seed,
        "selection_metric": "f1@iou0.5",
    }
    if exemplar_counts is None:
        # Flat, committed single-count shape (byte-identical to before this option existed).
        report["exemplar_count"] = exemplar_count
        report["methods"] = _methods_at(exemplar_count)
    else:
        # Nested per-count blocks: one tune-freeze-report per operating point.
        counts = tuple(exemplar_counts)
        report["exemplar_counts"] = list(counts)
        report["per_count"] = [
            {"exemplar_count": count, "methods": _methods_at(count)} for count in counts
        ]

    if out is not None:
        out_path = Path(out)
        if not out_path.is_absolute():
            out_path = repo_root() / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        logger.info("tuning: wrote {}", out_path)
    return report
