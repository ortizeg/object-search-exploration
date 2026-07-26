"""Tests for Method 1 (`ncc`).

These pin the behaviours the research measured and the plan calls mandatory: the textureless
guard, the no-duplicates lattice count, the exemplar self-label, the candidate/threshold split
(EVAL-08), and the OpenCV-version-specific flat-template constant so a silent 4->5 bump fails
loudly here instead of on the scoreboard.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from object_search.schemas import BBox, ExemplarBox, SearchOutcome
from object_search.search import get_method, has_method
from object_search.search.ncc import (
    _REPEAT_STRICT_FRAC,
    NCCConfig,
    _repeat_aware_threshold,
    search,
)
from object_search.synthetic.generator import DEMO_SPECS, synthesize

_NCC_SOURCE = Path(__file__).resolve().parents[1] / "src" / "object_search" / "search" / "ncc.py"


# --------------------------------------------------------------------------- source contract


def test_ncc_is_registered() -> None:
    assert has_method("ncc")
    spec = get_method("ncc")
    assert spec.config_model is NCCConfig
    assert spec.version == "1.0.0"


def test_source_carries_the_mandated_structure() -> None:
    text = _NCC_SOURCE.read_text(encoding="utf-8")
    assert "@register_method" in text
    assert "ROBUSTNESS BACKLOG" in text
    for step in range(1, 10):
        assert f"# {step}." in text, f"missing numbered step comment '# {step}.'"
    # METHOD-12: no single-best short-circuit. minMaxLoc must not be the peak path.
    assert "minMaxLoc" not in text


# ------------------------------------------------------------------------- textureless guard


def test_flat_crop_returns_empty_with_a_note() -> None:
    scene = np.full((200, 300, 3), 128, np.uint8)  # uniform => zero-variance crop
    exemplar = ExemplarBox(box=BBox(x=40, y=40, w=60, h=60))

    result = search(scene, exemplar, NCCConfig())

    assert result.outcome is SearchOutcome.EMPTY
    assert result.matches == ()
    assert result.diagnostics.notes, "an abstention must say why (METHOD-04c)"
    assert "texture" in result.diagnostics.notes[0].lower()


def test_flat_template_matchtemplate_behaviour_is_pinned() -> None:
    """Pin the measured flat-template response so an OpenCV change is caught (PITFALLS 1.1).

    A zero-variance template makes TM_CCOEFF_NORMED degenerate (0/0). OpenCV does not raise and
    does not return NaN -- it returns a CONSTANT map whose value is a **fabricated** degenerate
    constant, never a real correlation. *Which* constant appears is both version- AND
    platform-dependent, which is exactly why this test does not assert one specific number:

    - OpenCV 4.10 returned 1.0 everywhere (issue #5688);
    - the pinned 4.13 returns 0.0 on macOS/arm64 but 1.0 on Linux/x86_64 (measured: this test
      first failed in CI on Linux while passing locally on macOS -- the constant is a build
      artefact, not a documented value);
    - 5.0 returns 0.0.

    So the portable, honest invariants are pinned instead: the map is CONSTANT, carries no
    NaN/Inf, and its value is one of the two known degenerate constants {0.0, 1.0} -- never a
    plausible-looking correlation. Any of these breaking (a real value, a NaN, a non-constant
    map) fails loudly here rather than on the scoreboard. All of them are why :func:`search`'s
    std guard runs *before* matchTemplate regardless of which constant this build produces.
    """
    rng = np.random.default_rng(0)
    scene = rng.integers(0, 256, (120, 160), dtype=np.uint8)
    flat_template = np.full((20, 20), 100, np.uint8)

    resp = cv2.matchTemplate(scene, flat_template, cv2.TM_CCOEFF_NORMED)

    assert np.isfinite(resp).all(), "flat template must not produce NaN/Inf (it never has)"
    assert float(np.ptp(resp)) == 0.0, "a zero-variance template yields a constant response map"
    assert float(resp.flat[0]) in (0.0, 1.0), "the constant must be a known degenerate value"


# ------------------------------------------------------------------------- lattice behaviour


def _lattice() -> tuple[np.ndarray, tuple[BBox, ...]]:
    generated = synthesize(DEMO_SPECS["lattice-plain"])
    return generated.image, generated.boxes


def test_lattice_returns_every_instance_without_duplicates() -> None:
    scene, boxes = _lattice()
    exemplar = ExemplarBox(box=boxes[0])

    result = search(scene, exemplar, NCCConfig())

    assert result.outcome is SearchOutcome.OK
    # Success criterion 1: every instance, no duplicates.
    assert len(result.matches) == len(boxes)
    # No two accepted boxes overlap beyond the NMS threshold -> no duplicate detections.
    for i in range(len(result.matches)):
        for j in range(i + 1, len(result.matches)):
            iou = result.matches[i].box.iou(result.matches[j].box)
            assert iou < NCCConfig().nms_iou, f"matches {i},{j} overlap (IoU={iou:.3f})"


def test_never_short_circuits_to_a_single_best() -> None:
    scene, boxes = _lattice()
    exemplar = ExemplarBox(box=boxes[0])

    result = search(scene, exemplar, NCCConfig())

    # METHOD-12: a lattice of N instances returns N matches, not 1.
    assert len(result.matches) > 1


def test_exemplar_region_is_labelled_is_exemplar() -> None:
    scene, boxes = _lattice()
    exemplar = ExemplarBox(box=boxes[0])

    result = search(scene, exemplar, NCCConfig())

    flagged = [m for m in result.matches if m.is_exemplar]
    assert len(flagged) == 1, "exactly one match is the exemplar's own region"
    assert flagged[0].box.iou(boxes[0]) >= 0.5


def test_matches_and_candidates_both_present() -> None:
    scene, boxes = _lattice()
    exemplar = ExemplarBox(box=boxes[0])

    result = search(scene, exemplar, NCCConfig())

    # The candidate log (EVAL-08) carries raw scores and is at least as rich as the match set.
    assert result.threshold_applied is not None
    assert len(result.candidates) >= len(result.matches)
    assert all(isinstance(c.score, float) for c in result.candidates)


# --------------------------------------------------- candidate/threshold split (EVAL-08 core)


def test_candidates_exceed_matches_when_threshold_excludes_a_peak() -> None:
    """A sub-threshold-but-real peak must survive as a Candidate while being excluded as a Match."""
    rng = np.random.default_rng(1)
    scene = np.full((300, 500, 3), 200, np.uint8)
    patch = rng.integers(0, 256, (60, 60, 3), dtype=np.uint8)  # strong texture => sharp peak
    scene[40:100, 40:100] = patch  # exemplar
    scene[40:100, 240:300] = patch  # identical clean copy => raw ~1.0
    noisy = np.clip(patch.astype(np.int64) + rng.normal(0, 70, (60, 60, 3)), 0, 255).astype(
        np.uint8
    )
    scene[180:240, 40:100] = noisy  # degraded copy => raw well below 1.0

    exemplar = ExemplarBox(box=BBox(x=40, y=40, w=60, h=60))
    config = NCCConfig(scales=(1.0,), threshold=0.85, calibration="fixed")

    result = search(scene, exemplar, config)

    assert result.outcome is SearchOutcome.OK
    assert result.threshold_applied == pytest.approx(0.85)
    # The clean pair clears 0.85; the noisy copy does not.
    assert all(m.score > 0.85 for m in result.matches)
    assert len(result.candidates) > len(result.matches)
    assert any(c.score <= 0.85 for c in result.candidates), (
        "the excluded peak must remain a candidate"
    )


def test_impossible_threshold_yields_empty_not_error() -> None:
    scene, boxes = _lattice()
    exemplar = ExemplarBox(box=boxes[0])

    result = search(scene, exemplar, NCCConfig(threshold=1.5, calibration="fixed"))

    assert result.outcome is SearchOutcome.EMPTY
    assert result.matches == ()
    assert result.threshold_applied == pytest.approx(1.5)
    assert result.diagnostics.notes


# ------------------------------------------------------------- strategy / calibration coverage


@pytest.mark.parametrize("peaks_strategy", ["nms", "local-max", "watershed"])
def test_every_peak_strategy_runs(peaks_strategy: str) -> None:
    scene, boxes = _lattice()
    exemplar = ExemplarBox(box=boxes[0])

    result = search(scene, exemplar, NCCConfig(peaks=peaks_strategy))  # type: ignore[arg-type]

    assert result.outcome in {SearchOutcome.OK, SearchOutcome.EMPTY}


@pytest.mark.parametrize("strategy", ["self-similarity", "ratio", "gmm"])
def test_every_calibration_strategy_runs(strategy: str) -> None:
    scene, boxes = _lattice()
    exemplar = ExemplarBox(box=boxes[0])

    result = search(scene, exemplar, NCCConfig(calibration=strategy))  # type: ignore[arg-type]

    assert result.threshold_applied is not None


def test_rotation_bank_runs_with_masked_correlation() -> None:
    scene, boxes = _lattice()
    exemplar = ExemplarBox(box=boxes[0])

    # A non-zero angle exercises the warped-template + eroded-mask correlation path.
    result = search(scene, exemplar, NCCConfig(angles_deg=(0.0, 15.0)))

    assert result.outcome is SearchOutcome.OK
    assert len(result.matches) >= 1


# --------------------------------------------------------------- rotation + repeat-aware defaults


def test_default_config_enables_rotation_bank_and_repeat_aware() -> None:
    """The shipped default recovers rotated repeats: a multi-angle bank + repeat-aware cut."""
    config = NCCConfig()
    # A rotation bank (more than the single 0 deg angle) is on by default now.
    assert config.angles_deg.count(0.0) == 1
    assert len(config.angles_deg) >= 5, "rotation bank should span several angles by default"
    assert max(abs(a) for a in config.angles_deg) >= 30.0
    assert config.calibration == "repeat-aware"


def test_repeat_aware_is_strict_on_near_identical_repeats() -> None:
    """Two+ distinct near-self locations => cut just below the self cluster (reject rot FPs)."""
    calib = _repeat_aware_threshold(5, self_score=1.0, retain_frac=0.45)
    assert calib.strategy == "repeat-aware"
    assert calib.threshold == pytest.approx(_REPEAT_STRICT_FRAC)  # self * strict, self==1.0
    assert "near-identical" in calib.reason


def test_repeat_aware_is_permissive_when_instances_are_transformed() -> None:
    """Only the exemplar itself sits near self => permissive self*retain_frac tail for copies."""
    calib = _repeat_aware_threshold(1, self_score=1.0, retain_frac=0.45)
    assert calib.threshold == pytest.approx(0.45)
    assert "transformed" in calib.reason
    # The strict cut is always higher than the permissive one, so a repeat set is never *more*
    # permissive than a transformed set (the whole point of the switch).
    assert _repeat_aware_threshold(3, 1.0, retain_frac=0.45).threshold > calib.threshold


def test_candidate_log_is_deduplicated_across_the_bank() -> None:
    """The pyramid x rotation bank must not enter one instance into the candidate log many times."""
    scene, boxes = _lattice()
    exemplar = ExemplarBox(box=boxes[0])

    result = search(scene, exemplar, NCCConfig())

    iou = NCCConfig().nms_iou
    # No two candidates are duplicates of one location ...
    for i in range(len(result.candidates)):
        for j in range(i + 1, len(result.candidates)):
            assert result.candidates[i].box.iou(result.candidates[j].box) < iou
    # ... and no candidate duplicates an accepted match (so matches + candidates is one clean set).
    for cand in result.candidates:
        for match in result.matches:
            assert cand.box.iou(match.box) < iou


def test_diagnostics_carry_heatmap_and_metrics() -> None:
    scene, boxes = _lattice()
    exemplar = ExemplarBox(box=boxes[0])

    result = search(scene, exemplar, NCCConfig())

    assert result.diagnostics.similarity_heatmap is not None
    assert result.diagnostics.metrics["crop_std"] > 0.0
    assert "n_matches" in result.diagnostics.metrics
    assert result.latency.total_ms >= 0.0
