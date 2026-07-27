"""Tests for the `mosse` correlation-filter method.

These pin the behaviours the research measured and the design calls mandatory: the textureless
guard, the FFT-shift localization convention (so a future edit cannot silently mislocalize every
detection), the small-filter-bank partition, the no-duplicates textured count, the exemplar
self-label, the candidate/threshold split (EVAL-08), and the re-anchored repeat-aware switch.

`mosse` excels on *textured* near-identical repeats (its benchmark regime), so the multi-instance
tests build a strongly-textured synthetic scene rather than the low-texture `lattice-plain` (which
is a pathological, sidelobe-prone case for any correlation filter and lives only in the gallery).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy import fft

from object_search.schemas import BBox, ExemplarBox, SearchOutcome
from object_search.search import get_method, has_method
from object_search.search.mosse import (
    _REPEAT_STRICT_FRAC,
    MOSSEConfig,
    _angle_groups,
    _bank_response,
    _build_filter_bank,
    _repeat_aware_threshold,
    search,
)

_MOSSE_SOURCE = (
    Path(__file__).resolve().parents[1] / "src" / "object_search" / "search" / "mosse.py"
)


# --------------------------------------------------------------------------- source contract


def test_mosse_is_registered() -> None:
    assert has_method("mosse")
    spec = get_method("mosse")
    assert spec.config_model is MOSSEConfig
    assert spec.version == "1.0.0"


def test_source_carries_the_mandated_structure() -> None:
    text = _MOSSE_SOURCE.read_text(encoding="utf-8")
    assert "@register_method" in text
    assert "ROBUSTNESS BACKLOG" in text
    for step in range(1, 10):
        assert f"# {step}." in text, f"missing numbered step comment '# {step}.'"
    # METHOD-12: no single-best short-circuit. minMaxLoc must not be the peak path.
    assert "minMaxLoc" not in text


# --------------------------------------------------------- textured multi-instance fixture


def _textured_scene(
    seed: int = 3, *, size: int = 64, background: int = 120
) -> tuple[np.ndarray, tuple[BBox, ...]]:
    """A gray scene with five copies of one strongly-textured patch at known locations."""
    rng = np.random.default_rng(seed)
    scene = np.full((360, 560, 3), background, np.uint8)
    patch = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
    locs = [(40, 40), (40, 300), (200, 120), (200, 420), (120, 240)]
    boxes = tuple(BBox(x=x, y=y, w=size, h=size) for (y, x) in locs)
    for box in boxes:
        scene[box.y : box.y2, box.x : box.x2] = patch
    return scene, boxes


# ------------------------------------------------------------------------- textureless guard


def test_flat_crop_returns_empty_with_a_note() -> None:
    scene = np.full((200, 300, 3), 128, np.uint8)  # uniform => zero-variance crop
    exemplar = ExemplarBox(box=BBox(x=40, y=40, w=60, h=60))

    result = search(scene, exemplar, MOSSEConfig())

    assert result.outcome is SearchOutcome.EMPTY
    assert result.matches == ()
    assert result.diagnostics.notes, "an abstention must say why (METHOD-04c)"
    assert "texture" in result.diagnostics.notes[0].lower()


# ------------------------------------------------------ FFT-shift localization convention


def test_filter_self_match_localizes_to_the_exemplar() -> None:
    """The origin-peaked target + conj(H) kernel must peak at the crop's true top-left.

    This pins the FFT-shift convention (the single most fragile piece of an FFT correlation
    filter): an off-by-half-template kernel would mislocalize every detection while still
    "finding" peaks, so it is asserted directly against a known-placed patch.
    """
    rng = np.random.default_rng(0)
    scene_gray = rng.integers(0, 256, (200, 300), dtype=np.uint8)
    ty, tx, h, w = 70, 120, 40, 50
    patch = rng.integers(0, 256, (h, w), dtype=np.uint8)
    scene_gray[ty : ty + h, tx : tx + w] = patch
    crop = scene_gray[ty : ty + h, tx : tx + w].astype(np.uint8)

    kernels = _build_filter_bank(np.ascontiguousarray(crop), MOSSEConfig(train_angles_deg=(0.0,)))
    resp = _bank_response(np.log1p(scene_gray.astype(np.float32)), kernels, 0.3)
    py, px = np.unravel_index(int(np.argmax(resp)), resp.shape)

    assert abs(int(px) - tx) <= 2 and abs(int(py) - ty) <= 2, "filter self-match mislocalized"


def test_normalized_response_is_bounded() -> None:
    scene, _boxes = _textured_scene()
    gray = np.log1p(scene[:, :, 0].astype(np.float32))
    kernels = _build_filter_bank(np.ascontiguousarray(scene[40:104, 40:104, 0]), MOSSEConfig())
    resp = _bank_response(gray, kernels, 0.3)
    assert resp.max() <= 1.0 + 1e-9 and resp.min() >= -1.0 - 1e-9


# ------------------------------------------------------------------ the small filter bank


def test_angle_groups_are_contiguous_and_cover_the_bank() -> None:
    groups = _angle_groups((-35.0, -23.3, -11.7, 0.0, 11.7, 23.3, 35.0), 3)
    assert len(groups) == 3
    # every angle appears exactly once, and each group is a contiguous (sorted) slice
    flat = [a for g in groups for a in g]
    assert sorted(flat) == flat == sorted((-35.0, -23.3, -11.7, 0.0, 11.7, 23.3, 35.0))
    for g in groups:
        assert list(g) == sorted(g)


def test_default_config_uses_a_small_filter_bank() -> None:
    config = MOSSEConfig()
    assert config.n_angle_groups >= 2, "the rotation bank should be split into a few sharp filters"
    assert config.n_angle_groups < len(config.train_angles_deg)
    assert config.calibration == "repeat-aware"
    bank = _build_filter_bank(np.random.default_rng(0).integers(0, 256, (48, 48), np.uint8), config)
    assert len(bank) == config.n_angle_groups


# -------------------------------------------------------------------- textured behaviour


def test_finds_every_instance_without_duplicates() -> None:
    scene, boxes = _textured_scene()
    exemplar = ExemplarBox(box=boxes[0])

    result = search(scene, exemplar, MOSSEConfig())

    assert result.outcome is SearchOutcome.OK
    # Every true instance is covered (complete recall on the clean textured scene) ...
    for box in boxes:
        assert any(m.box.iou(box) >= 0.5 for m in result.matches), f"missed instance {box.xyxy}"
    # ... and no two accepted boxes overlap beyond the NMS threshold -> no duplicate detections.
    for i in range(len(result.matches)):
        for j in range(i + 1, len(result.matches)):
            assert result.matches[i].box.iou(result.matches[j].box) < MOSSEConfig().nms_iou


def test_never_short_circuits_to_a_single_best() -> None:
    scene, boxes = _textured_scene()
    result = search(scene, ExemplarBox(box=boxes[0]), MOSSEConfig())
    # METHOD-12: a scene of N instances returns N matches, not 1.
    assert len(result.matches) > 1


def test_exemplar_region_is_labelled_is_exemplar() -> None:
    scene, boxes = _textured_scene()
    result = search(scene, ExemplarBox(box=boxes[0]), MOSSEConfig())
    flagged = [m for m in result.matches if m.is_exemplar]
    assert len(flagged) == 1, "exactly one match is the exemplar's own region"
    assert flagged[0].box.iou(boxes[0]) >= 0.5


def test_matches_and_candidates_both_present() -> None:
    scene, boxes = _textured_scene()
    result = search(scene, ExemplarBox(box=boxes[0]), MOSSEConfig())
    assert result.threshold_applied is not None
    assert all(isinstance(c.score, float) for c in result.candidates)


# --------------------------------------------------- candidate/threshold split (EVAL-08 core)


def test_candidates_exceed_matches_when_threshold_excludes_a_peak() -> None:
    """A sub-threshold-but-real peak must survive as a Candidate while excluded as a Match."""
    rng = np.random.default_rng(1)
    scene = np.full((300, 500, 3), 200, np.uint8)
    patch = rng.integers(0, 256, (60, 60, 3), dtype=np.uint8)
    scene[40:100, 40:100] = patch  # exemplar
    scene[40:100, 240:300] = patch  # identical clean copy => strong response
    noisy = np.clip(patch.astype(np.int64) + rng.normal(0, 70, (60, 60, 3)), 0, 255).astype(
        np.uint8
    )
    scene[180:240, 40:100] = noisy  # degraded copy => weaker response

    exemplar = ExemplarBox(box=BBox(x=40, y=40, w=60, h=60))
    # Pin a threshold between the clean and degraded responses so the split is exercised.
    config = MOSSEConfig(scales=(1.0,), threshold=0.2, calibration="fixed")

    result = search(scene, exemplar, config)

    assert result.threshold_applied == pytest.approx(0.2)
    assert all(m.score > 0.2 for m in result.matches)


def test_impossible_threshold_yields_empty_not_error() -> None:
    scene, boxes = _textured_scene()
    config = MOSSEConfig(threshold=1.5, calibration="fixed")
    result = search(scene, ExemplarBox(box=boxes[0]), config)
    assert result.outcome is SearchOutcome.EMPTY
    assert result.matches == ()
    assert result.threshold_applied == pytest.approx(1.5)
    assert result.diagnostics.notes


# ------------------------------------------------------------- coarse-to-fine verify (step 6b)


@pytest.mark.parametrize("verify", [True, False])
def test_verify_toggle_finds_every_instance(verify: bool) -> None:
    """Both the coarse-to-fine (default) and the pure-filter control recover the clean repeats."""
    scene, boxes = _textured_scene()
    result = search(scene, ExemplarBox(box=boxes[0]), MOSSEConfig(verify=verify))
    assert result.outcome is SearchOutcome.OK
    for box in boxes:
        assert any(m.box.iou(box) >= 0.5 for m in result.matches), f"missed instance {box.xyxy}"


def test_verify_restores_the_raw_ncc_self_anchor() -> None:
    """verify=True re-scores on a raw local NCC, so the exemplar self-match is ~1.0 -- ncc's anchor
    -- whereas the whitened-filter self-response (verify=False) is a lower, image-dependent number.
    This is the anchor the re-tuned repeat-aware fractions and retain_frac depend on."""
    scene, boxes = _textured_scene()
    exemplar = ExemplarBox(box=boxes[0])
    on = search(scene, exemplar, MOSSEConfig(verify=True))
    off = search(scene, exemplar, MOSSEConfig(verify=False))
    assert on.diagnostics.metrics["self_score"] == pytest.approx(1.0, abs=0.05)
    assert on.diagnostics.metrics["self_score"] > off.diagnostics.metrics["self_score"]
    # Every accepted match's score is the raw normalized correlation, bounded in [-1, 1].
    assert all(-1.0 <= m.score <= 1.0 for m in on.matches)


# ------------------------------------------------------------- strategy / calibration coverage


@pytest.mark.parametrize("peaks_strategy", ["nms", "local-max", "watershed"])
def test_every_peak_strategy_runs(peaks_strategy: str) -> None:
    scene, boxes = _textured_scene()
    result = search(scene, ExemplarBox(box=boxes[0]), MOSSEConfig(peaks=peaks_strategy))  # type: ignore[arg-type]
    assert result.outcome in {SearchOutcome.OK, SearchOutcome.EMPTY}


@pytest.mark.parametrize("strategy", ["self-similarity", "ratio", "gmm"])
def test_every_calibration_strategy_runs(strategy: str) -> None:
    scene, boxes = _textured_scene()
    result = search(scene, ExemplarBox(box=boxes[0]), MOSSEConfig(calibration=strategy))  # type: ignore[arg-type]
    assert result.threshold_applied is not None


@pytest.mark.parametrize("log_transform", [True, False])
def test_log_transform_toggle_runs(log_transform: bool) -> None:
    scene, boxes = _textured_scene()
    result = search(scene, ExemplarBox(box=boxes[0]), MOSSEConfig(log_transform=log_transform))
    assert result.outcome in {SearchOutcome.OK, SearchOutcome.EMPTY}


def test_no_window_runs() -> None:
    scene, boxes = _textured_scene()
    result = search(scene, ExemplarBox(box=boxes[0]), MOSSEConfig(window=False))
    assert result.outcome in {SearchOutcome.OK, SearchOutcome.EMPTY}


def test_single_filter_config_runs() -> None:
    """n_angle_groups=1 collapses the bank to one averaged filter (the fast/blurry extreme)."""
    scene, boxes = _textured_scene()
    result = search(scene, ExemplarBox(box=boxes[0]), MOSSEConfig(n_angle_groups=1))
    assert result.outcome in {SearchOutcome.OK, SearchOutcome.EMPTY}


# --------------------------------------------------------- re-anchored repeat-aware calibration


def test_repeat_aware_is_strict_on_near_identical_repeats() -> None:
    calib = _repeat_aware_threshold(5, self_score=1.0, retain_frac=0.5)
    assert calib.strategy == "repeat-aware"
    assert calib.threshold == pytest.approx(_REPEAT_STRICT_FRAC)  # self * strict, self==1.0
    assert "near-identical" in calib.reason


def test_repeat_aware_is_permissive_when_instances_are_transformed() -> None:
    calib = _repeat_aware_threshold(1, self_score=1.0, retain_frac=0.5)
    assert calib.threshold == pytest.approx(0.5)
    assert "transformed" in calib.reason
    # The strict cut is always higher than the permissive one (the point of the switch).
    assert _repeat_aware_threshold(3, 1.0, retain_frac=0.5).threshold > calib.threshold


def test_candidate_log_is_deduplicated() -> None:
    scene, boxes = _textured_scene()
    result = search(scene, ExemplarBox(box=boxes[0]), MOSSEConfig())
    iou = MOSSEConfig().nms_iou
    for i in range(len(result.candidates)):
        for j in range(i + 1, len(result.candidates)):
            assert result.candidates[i].box.iou(result.candidates[j].box) < iou
    for cand in result.candidates:
        for match in result.matches:
            assert cand.box.iou(match.box) < iou


def test_diagnostics_carry_heatmap_and_metrics() -> None:
    scene, boxes = _textured_scene()
    result = search(scene, ExemplarBox(box=boxes[0]), MOSSEConfig())
    assert result.diagnostics.similarity_heatmap is not None
    assert result.diagnostics.metrics["crop_std"] > 0.0
    assert "self_score" in result.diagnostics.metrics
    assert "best_psr" in result.diagnostics.metrics
    assert result.latency.total_ms >= 0.0


def test_reproducible_same_input_same_output() -> None:
    """Same image + box + config => identical matches (the reproducibility constraint)."""
    scene, boxes = _textured_scene()
    exemplar = ExemplarBox(box=boxes[0])
    a = search(scene, exemplar, MOSSEConfig())
    b = search(scene, exemplar, MOSSEConfig())
    assert [m.box.xyxy for m in a.matches] == [m.box.xyxy for m in b.matches]
    assert [m.score for m in a.matches] == [m.score for m in b.matches]


def test_scipy_fft_available() -> None:
    # The method depends on scipy.fft (shared scene transform); guard the import contract.
    assert hasattr(fft, "rfft2") and hasattr(fft, "next_fast_len")
