"""Tests for Method 3 (``dino-dense``).

Two tiers, deliberately (mirroring ``test_dinov2.py``):

* **Model-free logic** -- the L2-normalization order, the cosine-not-raw-dot similarity, the
  map upsampling alignment, the connected-components label-0 skip, the calibration comparison,
  the config schema, and the model-absent error path. These need no weight and so **run in CI**,
  gating the risky post-processing logic.
* **Real-model behaviour** -- the end-to-end search, the resolution cap, and the headline
  success criterion (dino-dense beats ncc on pose variation). These need the gitignored
  ``dinov2_small.onnx`` and are **skipped when the weight is absent** (CI cannot fetch it).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
import pytest
from pydantic import ValidationError

from object_search.inference import models
from object_search.schemas import BBox, ExemplarBox, Match, SearchOutcome
from object_search.search import dino_dense, has_method
from object_search.search.common import calibration
from object_search.search.dino_dense import (
    DinoDenseConfig,
    _contrast_threshold,
    _crop_token_bank,
    _extract_components,
    _l2_normalize,
    _maxtoken_similarity_map,
    _prototype_from_grid,
    _similarity_map,
    _upsample_similarity,
    reset_inferencer_cache,
    search,
)
from object_search.search.ncc import NCCConfig
from object_search.search.ncc import search as ncc_search

_MODEL_PATH: Path = models.models_dir() / models.MODEL_REGISTRY["dinov2-small"].dest
_HAVE_MODEL: bool = _MODEL_PATH.is_file()
_needs_model = pytest.mark.skipif(
    not _HAVE_MODEL,
    reason=f"dinov2-small weight absent at {_MODEL_PATH} (gitignored; run pixi run fetch-models)",
)
_DIM = 384


@pytest.fixture(autouse=True)
def _isolate_inferencer_cache() -> object:
    """Reset the module-level inferencer cache around every test so monkeypatching cannot leak."""
    reset_inferencer_cache()
    yield
    reset_inferencer_cache()


# ------------------------------------------------------------------- model-free: the config


def test_config_defaults_match_the_locked_decisions() -> None:
    cfg = DinoDenseConfig()
    assert cfg.scene_max_side == 1568
    # `contrast` + `max-token` are the defaults that lifted textured F1 from ~0.03 to ~0.70;
    # `gmm` (the old default) sat in the background shoulder and shipped a single full-frame box.
    assert cfg.calibration == "contrast"
    assert cfg.scoring == "max-token"
    assert cfg.match_tokens == 3
    assert cfg.threshold is None
    assert cfg.min_component_area == 4
    assert cfg.min_area_frac == pytest.approx(0.12)
    assert cfg.max_area_frac == pytest.approx(8.0)
    assert cfg.max_candidates == 50
    assert cfg.seed == 0


def test_config_is_frozen_and_schema_drives_the_form() -> None:
    cfg = DinoDenseConfig()
    with pytest.raises(ValidationError):  # frozen -> mutation is an error
        cfg.threshold = 0.5  # type: ignore[misc]
    schema = DinoDenseConfig.model_json_schema()
    # Every field carries a description (it becomes the UI form's help string).
    for field in ("scene_max_side", "calibration", "threshold", "min_component_area"):
        assert schema["properties"][field].get("description")


def test_registered_under_dino_dense_with_its_config() -> None:
    assert has_method("dino-dense")
    from object_search.search import get_method

    spec = get_method("dino-dense")
    assert spec.config_model is DinoDenseConfig


# ------------------------------------------------- model-free: L2-normalization order (a truth)


def test_l2_normalize_makes_unit_vectors_and_keeps_zero_zero() -> None:
    vecs = np.array([[3.0, 4.0], [0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    out = _l2_normalize(vecs, axis=1)
    norms = np.linalg.norm(out, axis=1)
    assert norms[0] == pytest.approx(1.0)  # (3,4) -> unit
    assert norms[1] == pytest.approx(0.0)  # zero stays zero, never NaN
    assert norms[2] == pytest.approx(1.0)
    assert np.isfinite(out).all()


def test_prototype_is_mean_pooled_then_l2_normalized() -> None:
    """Order is load-bearing: pool the tokens FIRST, then normalize once (self-cosine 1.0)."""
    rng = np.random.default_rng(0)
    crop_grid = rng.standard_normal((4, 5, _DIM)).astype(np.float32)
    proto = _prototype_from_grid(crop_grid)
    # It is a unit vector.
    assert np.linalg.norm(proto) == pytest.approx(1.0, abs=1e-5)
    # It is the normalized mean, not the mean of normalized tokens.
    manual = crop_grid.reshape(-1, _DIM).mean(axis=0)
    manual = manual / np.linalg.norm(manual)
    assert np.allclose(proto, manual, atol=1e-5)


def test_similarity_is_cosine_not_magnitude_dominated_raw_dot() -> None:
    """A 5x-magnitude token and a unit token in the prototype direction must both score ~1.0.

    A raw (unnormalized) dot product would rank the 5x token far above the unit one; cosine
    treats them equally. This is the exact bug the L2-normalization prevents.
    """
    u = np.zeros(_DIM, dtype=np.float32)
    u[0] = 1.0  # prototype direction (already unit)
    orth = np.zeros(_DIM, dtype=np.float32)
    orth[1] = 9.0  # large but orthogonal
    grid = np.stack([5.0 * u, 1.0 * u, orth]).reshape(1, 3, _DIM).astype(np.float32)
    sim = _similarity_map(grid, u)
    assert sim.shape == (1, 3)
    assert sim[0, 0] == pytest.approx(1.0, abs=1e-5)  # 5x magnitude, same direction -> cosine 1
    assert sim[0, 1] == pytest.approx(1.0, abs=1e-5)  # unit, same direction -> cosine 1
    assert sim[0, 2] == pytest.approx(0.0, abs=1e-5)  # orthogonal -> cosine 0


# ---------------------------------------------------- model-free: MAP upsampling alignment


def test_upsample_aligns_a_known_token_peak_to_its_patch_pixel() -> None:
    """A one-hot token at ``(gy, gx)`` must upsample to a peak at that patch's centre pixel."""
    gh, gw = 32, 64
    gy, gx = 3, 50  # deliberately gy != gx and gx > gh-1 so a transpose could never pass
    sim = np.zeros((gh, gw), dtype=np.float32)
    sim[gy, gx] = 1.0
    # scale factors of 1.0 mean the input was already a multiple of 14 (no snap).
    full = _upsample_similarity(sim, scale_x=1.0, scale_y=1.0)
    assert full.shape == (gh * 14, gw * 14)
    py, px = np.unravel_index(int(np.argmax(full)), full.shape)
    patch = dino_dense.DINOV2_PATCH
    assert abs(int(px) - (gx * patch + patch // 2)) <= 1  # patch centre in x
    assert abs(int(py) - (gy * patch + patch // 2)) <= 1  # patch centre in y


def test_upsample_uses_the_scale_factors_to_recover_input_size() -> None:
    """Non-unit scale factors invert the snap: target size = grid*14 / scale."""
    gh, gw = 16, 23
    sim = np.zeros((gh, gw), dtype=np.float32)
    sim[5, 10] = 1.0
    scale_x, scale_y = 322 / 320, 224 / 225  # from a 320x225 scene snapped to 322x224
    full = _upsample_similarity(sim, scale_x=scale_x, scale_y=scale_y)
    assert full.shape == (round(gh * 14 / scale_y), round(gw * 14 / scale_x))
    # The peak lands at the patch centre expressed in the recovered input pixels.
    py, px = np.unravel_index(int(np.argmax(full)), full.shape)
    assert abs(int(px) - round((10 * 14 + 7) / scale_x)) <= 2
    assert abs(int(py) - round((5 * 14 + 7) / scale_y)) <= 2


# ------------------------------------------- model-free: connected components skip label 0


def _map_with_two_blobs() -> npt.NDArray[np.float32]:
    sim = np.full((300, 400), 0.1, dtype=np.float32)
    sim[40:70, 50:80] = 0.9  # blob A
    sim[200:230, 300:340] = 0.9  # blob B
    return sim


def test_extract_components_skips_label_zero_background() -> None:
    """The background is connected-components label 0 and must NOT be emitted as a box."""
    sim = _map_with_two_blobs()
    comps = _extract_components(
        sim, floor=0.5, min_area=4, max_area=0.0, cap_scale=1.0, orig_w=400, orig_h=300
    )
    # Exactly the two foreground blobs -- not three (the background label 0 is skipped).
    assert len(comps) == 2
    # No component spans the whole image (which is what emitting label 0 would look like).
    for comp in comps:
        assert comp.box.w < 100 and comp.box.h < 100
    centres = sorted((round(c.box.cx), round(c.box.cy)) for c in comps)
    assert centres[0] == pytest.approx((64, 54), abs=3)  # blob A centre
    assert centres[1] == pytest.approx((319, 214), abs=3)  # blob B centre


def test_extract_components_respects_min_area_and_cap_scale() -> None:
    sim = np.full((100, 100), 0.1, dtype=np.float32)
    sim[10:12, 10:12] = 0.9  # a 2x2 = area-4 blob
    sim[50, 50] = 0.9  # a 1-pixel blob (area 1) -- below min_area
    comps = _extract_components(
        sim, floor=0.5, min_area=4, max_area=0.0, cap_scale=1.0, orig_w=100, orig_h=100
    )
    assert len(comps) == 1  # the 1-pixel blob was dropped
    # cap_scale halves capped-pixel coordinates back to original pixels.
    comps2 = _extract_components(
        sim, floor=0.5, min_area=4, max_area=0.0, cap_scale=2.0, orig_w=100, orig_h=100
    )
    assert comps2[0].box.x == round(10 / 2.0)


def test_extract_components_drops_oversized_merged_blobs() -> None:
    """A component above ``max_area`` is a merged/background blob and must be dropped."""
    sim = np.full((100, 100), 0.1, dtype=np.float32)
    sim[10:14, 10:14] = 0.9  # a 4x4 = area-16 instance-sized blob (kept)
    sim[40:90, 20:90] = 0.9  # a 50x70 = area-3500 merged/background blob (dropped by ceiling)
    comps = _extract_components(
        sim, floor=0.5, min_area=4, max_area=100.0, cap_scale=1.0, orig_w=100, orig_h=100
    )
    assert len(comps) == 1  # only the small blob survives the [4, 100] area window
    assert comps[0].box.w < 10 and comps[0].box.h < 10


def test_maxtoken_similarity_is_high_contrast_versus_prototype() -> None:
    """max-token scoring separates a matching token from background more than the mean-pool dot.

    Build a crop bank of two distinct part-vectors and a scene where one token equals part A,
    one equals part B, and the rest are an unrelated background direction. The best-part score is
    ~1 on the two instance tokens and low on background -- a wider gap than the prototype dot,
    whose mean-of-two-parts vector only half-matches each true part.
    """
    part_a = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    part_b = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    background = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    crop_grid = np.stack([part_a, part_b]).reshape(1, 2, 4)
    bank = _crop_token_bank(crop_grid)
    scene = np.stack([part_a, part_b, background, background]).reshape(2, 2, 4)

    max_map = _maxtoken_similarity_map(scene, bank, match_tokens=1)
    proto_map = _similarity_map(scene, _prototype_from_grid(crop_grid))
    # max-token: the two instance tokens score ~1, background ~0.
    assert max_map.reshape(-1)[0] == pytest.approx(1.0, abs=1e-5)
    assert max_map.reshape(-1)[1] == pytest.approx(1.0, abs=1e-5)
    assert max_map.reshape(-1)[2] == pytest.approx(0.0, abs=1e-5)
    # The instance-vs-background contrast is strictly larger for max-token than for the prototype.
    max_gap = float(max_map.reshape(-1)[0] - max_map.reshape(-1)[2])
    proto_gap = float(proto_map.reshape(-1)[0] - proto_map.reshape(-1)[2])
    assert max_gap > proto_gap


def test_contrast_threshold_sits_between_background_bulk_and_foreground_tail() -> None:
    """The contrast cut lands above the background bulk and below the instance tail."""
    rng = np.random.default_rng(0)
    background = rng.normal(0.30, 0.03, 2000)
    foreground = rng.normal(0.80, 0.02, 60)  # a thin high tail, as real instances are
    sim = np.concatenate([background, foreground]).astype(np.float32)
    threshold, reason = _contrast_threshold(sim)
    assert 0.35 < threshold < 0.80  # above the bulk, below the peak
    assert float(background.mean()) < threshold < float(foreground.mean())
    assert "blend" in reason and "cut at" in reason


# ------------------------------------------------ model-free: the three strategies differ


def test_three_calibration_strategies_yield_different_reasoned_thresholds() -> None:
    """Phase 6 success criterion 3: self-similarity, ratio, gmm cut differently on one map."""
    rng = np.random.default_rng(0)
    scores = np.concatenate([rng.normal(0.20, 0.02, 500), rng.normal(0.70, 0.02, 120)]).astype(
        np.float64
    )

    self_sim = calibration.calibrate(scores, strategy="self-similarity", self_score=1.0)
    ratio = calibration.calibrate(scores, strategy="ratio")
    gmm = calibration.calibrate(scores, strategy="gmm", seed=0)

    thresholds = [self_sim.threshold, ratio.threshold, gmm.threshold]
    # All three are pairwise distinct (different, inspectable cuts).
    assert len(thresholds) == len({round(t, 4) for t in thresholds})
    # gmm cuts between the two modes; self-similarity anchors on the self score.
    assert 0.3 < gmm.threshold < 0.6
    assert self_sim.threshold == pytest.approx(0.7)
    for result in (self_sim, ratio, gmm):
        assert result.reason  # each carries its human-readable justification


# --------------------------------------------------- model-free: the model-absent error path


def test_search_returns_model_unavailable_error_when_weight_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no weight on disk the method degrades to outcome=error, never a raise."""
    monkeypatch.setattr(dino_dense.models, "models_dir", lambda: tmp_path)
    reset_inferencer_cache()  # force a re-probe against the empty temp dir
    scene = np.full((60, 60, 3), 50, dtype=np.uint8)
    result = search(scene, ExemplarBox(box=BBox(x=5, y=5, w=20, h=20)), DinoDenseConfig())
    assert result.outcome is SearchOutcome.ERROR
    assert result.error is not None
    assert result.error.kind == "model_unavailable"
    assert result.matches == ()


def test_search_rejects_a_foreign_config() -> None:
    scene = np.full((60, 60, 3), 50, dtype=np.uint8)
    with pytest.raises(TypeError, match="requires a DinoDenseConfig"):
        search(scene, ExemplarBox(box=BBox(x=5, y=5, w=20, h=20)), NCCConfig())


# --------------------------------- model-free: the full search path (stubbed inferencer)
#
# The dense-matching core of `search` -- scoring, upsampling, contrast calibration, connected-
# component extraction, exemplar labelling, the candidate/threshold split and diagnostics -- is
# reachable in CI with NO gitignored weight by injecting a stub through the `_get_inferencer`
# seam the module already exposes (the same pattern `test_owlv2_oneshot` uses). Without this, that
# whole path is exercised only by the real-weight tests below, which skip in CI.

_STUB_DIM = 8  # a tiny embedding: index 0 is the object direction, index 1 the background one.


class _StubDinoInferencer:
    """A minimal DINOv2 stand-in returning fixed token grids: object tokens vs background tokens.

    Proves ``search`` needs only something with a ``dense_tokens(image) -> (grid, sx, sy)`` method.
    The crop is far smaller than the scene, so image size disambiguates which grid to return --
    exactly how the real two-call path (encode the crop, then the scene) is driven.
    """

    def __init__(
        self, crop_grid: npt.NDArray[np.float32], scene_grid: npt.NDArray[np.float32]
    ) -> None:
        self.crop_grid = crop_grid
        self.scene_grid = scene_grid
        self.calls: list[tuple[int, int]] = []

    def dense_tokens(
        self, image: npt.NDArray[np.uint8]
    ) -> tuple[npt.NDArray[np.float32], float, float]:
        h, w = int(image.shape[0]), int(image.shape[1])
        self.calls.append((h, w))
        # scale_x/scale_y = 1.0: the stub grids already correspond 1:1 to a multiple-of-14 input,
        # so `_upsample_similarity` recovers the exact scene pixel size (gw*14 x gh*14).
        return (self.crop_grid, 1.0, 1.0) if max(h, w) <= 60 else (self.scene_grid, 1.0, 1.0)


# Four well-separated 3x3-token object blocks on a 10x15 background grid. At scale 1.0 that grid
# upsamples to a 140x210 scene; block (row, col) top-lefts are chosen so no two blocks touch after
# the bilinear ramp, so connected components emits four distinct instances (METHOD-12).
_STUB_BLOCKS: tuple[tuple[int, int], ...] = ((0, 0), (0, 12), (7, 0), (7, 6))
# The exemplar box covers the top-left block's pixel footprint (its object-token centres span
# x,y in [7, 35]); a generous 36px box overlaps that component well above the 0.5 exemplar IoU.
_STUB_EXEMPLAR = ExemplarBox(box=BBox(x=3, y=3, w=36, h=36))


def _stub_grids() -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Build the (crop_grid, scene_grid) pair: uniform-object crop, four object blocks on scene."""
    crop = np.zeros((3, 3, _STUB_DIM), dtype=np.float32)
    crop[..., 0] = 1.0  # every crop token points in the object direction -> prototype/bank = e0

    scene = np.zeros((10, 15, _STUB_DIM), dtype=np.float32)
    scene[..., 1] = 1.0  # background tokens are orthogonal to the object (cosine 0)
    for r0, c0 in _STUB_BLOCKS:
        scene[r0 : r0 + 3, c0 : c0 + 3, :] = 0.0
        scene[r0 : r0 + 3, c0 : c0 + 3, 0] = 1.0  # object tokens (cosine 1 with the crop bank)
    return crop, scene


def _stub_scene() -> npt.NDArray[np.uint8]:
    """A 140x210 scene image; its pixels are irrelevant (the stub keys off image size only)."""
    return np.zeros((140, 210, 3), dtype=np.uint8)


def test_search_end_to_end_with_a_stub_inferencer(monkeypatch: pytest.MonkeyPatch) -> None:
    """search embeds crop + scene, scores, calibrates, extracts components, labels the exemplar."""
    stub = _StubDinoInferencer(*_stub_grids())
    monkeypatch.setattr(dino_dense, "_get_inferencer", lambda: stub)

    result = search(_stub_scene(), _STUB_EXEMPLAR, DinoDenseConfig())  # default contrast/max-token

    assert result.outcome is SearchOutcome.OK
    # METHOD-12: four object blocks -> four matches, never a single-best short-circuit.
    assert len(result.matches) == 4
    assert sum(m.is_exemplar for m in result.matches) == 1  # the block overlapping the exemplar
    assert all(-1.0 <= m.score <= 1.0 for m in result.matches)  # scores are cosine similarities
    assert all(m.transform is None for m in result.matches)  # appearance method, no affine
    # Two forward passes, attributed separately: the crop encode, then the scene encode.
    assert stub.calls[0] == (36, 36) and stub.calls[1] == (140, 210)
    # Diagnostics carry the debug heatmap and the shipped metrics keys.
    assert result.diagnostics.similarity_heatmap is not None
    assert result.diagnostics.metrics["n_matches"] == 4.0
    assert result.diagnostics.metrics["grid_h"] == 10.0
    assert result.diagnostics.metrics["grid_w"] == 15.0
    assert result.diagnostics.metrics["sim_max"] == pytest.approx(1.0, abs=1e-5)
    assert result.candidates  # sub-threshold candidates retained (EVAL-08)


def test_search_prototype_scoring_path_finds_instances(monkeypatch: pytest.MonkeyPatch) -> None:
    """The `prototype` scoring branch (mean-pooled vector, not the token bank) recovers them too."""
    stub = _StubDinoInferencer(*_stub_grids())
    monkeypatch.setattr(dino_dense, "_get_inferencer", lambda: stub)

    result = search(_stub_scene(), _STUB_EXEMPLAR, DinoDenseConfig(scoring="prototype"))

    assert result.outcome is SearchOutcome.OK
    assert len(result.matches) == 4


@pytest.mark.parametrize("calibration_strategy", ["self-similarity", "ratio", "gmm"])
def test_search_every_delegated_calibration_runs(
    monkeypatch: pytest.MonkeyPatch, calibration_strategy: str
) -> None:
    """The classical calibrators (delegated to the shared `calibrate`) each produce a threshold."""
    stub = _StubDinoInferencer(*_stub_grids())
    monkeypatch.setattr(dino_dense, "_get_inferencer", lambda: stub)

    result = search(
        _stub_scene(),
        _STUB_EXEMPLAR,
        DinoDenseConfig(calibration=calibration_strategy),  # type: ignore[arg-type]
    )

    assert result.threshold_applied is not None
    assert result.outcome in {SearchOutcome.OK, SearchOutcome.EMPTY}


def test_search_empty_when_threshold_clears_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """An impossible fixed threshold yields EMPTY-with-note, never a raise (the no-match branch)."""
    stub = _StubDinoInferencer(*_stub_grids())
    monkeypatch.setattr(dino_dense, "_get_inferencer", lambda: stub)

    result = search(_stub_scene(), _STUB_EXEMPLAR, DinoDenseConfig(threshold=1.5))

    assert result.outcome is SearchOutcome.EMPTY
    assert result.matches == ()
    assert result.threshold_applied == pytest.approx(1.5)
    assert result.diagnostics.notes  # a note explains that nothing cleared the threshold


# ================================================== real-model behaviour (skipped in CI)


def _make_object(size: int, rng: np.random.Generator) -> npt.NDArray[np.uint8]:
    """A distinctive, textured, ASYMMETRIC colour motif so rotation truly changes the pixels."""
    obj = np.full((size, size, 3), 30, dtype=np.uint8)
    cv2.rectangle(obj, (6, 6), (size - 6, size // 2), (30, 60, 200), -1)
    cv2.circle(obj, (size // 3, 2 * size // 3), size // 5, (200, 180, 20), -1)
    cv2.line(obj, (4, size - 6), (size - 6, size - 10), (40, 220, 40), 5)
    for _ in range(40):
        x, y = rng.integers(0, size, 2)
        colour = tuple(int(v) for v in rng.integers(0, 255, 3))
        cv2.circle(obj, (int(x), int(y)), 2, colour, -1)
    return obj


def _pose_varied_scene() -> tuple[npt.NDArray[np.uint8], ExemplarBox, list[BBox]]:
    """One upright exemplar plus two ROTATED copies on a textured background.

    NCC's default bank is angles=(0,), so it cannot match the rotated copies; DINOv2 features are
    rotation-tolerant, so dino-dense can. The two rotated boxes are the non-exemplar ground truth.
    """
    rng = np.random.default_rng(0)
    height, width = 560, 840
    scene = np.full((height, width, 3), 40, dtype=np.uint8)
    for _ in range(400):
        x, y = rng.integers(0, width), rng.integers(0, height)
        colour = tuple(int(v) for v in rng.integers(0, 120, 3))
        cv2.circle(scene, (int(x), int(y)), 3, colour, -1)
    obj = _make_object(70, rng)
    scene[60:130, 60:130] = obj
    scene[350:420, 400:470] = cv2.rotate(obj, cv2.ROTATE_90_CLOCKWISE)
    scene[380:450, 200:270] = cv2.rotate(obj, cv2.ROTATE_180)
    exemplar = ExemplarBox(box=BBox(x=60, y=60, w=70, h=70))
    ground_truth = [BBox(x=400, y=350, w=70, h=70), BBox(x=200, y=380, w=70, h=70)]
    return scene, exemplar, ground_truth


def _covered(matches: tuple[Match, ...], instance: BBox, frac: float = 0.3) -> bool:
    """True if some match covers at least ``frac`` of the instance (robust to blob merging)."""
    for match in matches:
        box = match.box
        ix = max(box.x, instance.x)
        iy = max(box.y, instance.y)
        ix2 = min(box.x2, instance.x2)
        iy2 = min(box.y2, instance.y2)
        inter = max(0, ix2 - ix) * max(0, iy2 - iy)
        if inter / instance.area > frac:
            return True
    return False


def _recall(matches: tuple[Match, ...], ground_truth: list[BBox]) -> float:
    return sum(_covered(matches, gt) for gt in ground_truth) / len(ground_truth)


@_needs_model
def test_search_finds_multiple_instances_end_to_end() -> None:
    """METHOD-12: connected components returns many; never a single-best short-circuit."""
    scene, exemplar, _ = _pose_varied_scene()
    result = search(scene, exemplar, DinoDenseConfig())
    assert result.outcome is SearchOutcome.OK
    assert len(result.matches) > 1
    # The diagnostics carry the similarity heatmap the UI overlays.
    assert result.diagnostics.similarity_heatmap is not None
    assert result.candidates  # sub-threshold candidates retained (EVAL-08)
    # Exactly one match is the exemplar's own region.
    assert sum(m.is_exemplar for m in result.matches) <= 1


@_needs_model
def test_dino_dense_beats_ncc_on_pose_variation() -> None:
    """Phase 6 success criterion 1: dino-dense finds rotated instances that ncc misses.

    Verified, not asserted: the two methods genuinely disagree -- ncc's recall on the rotated
    copies is strictly lower than dino-dense's, and dino-dense covers at least one instance ncc
    does not.
    """
    scene, exemplar, ground_truth = _pose_varied_scene()

    dino = search(scene, exemplar, DinoDenseConfig())  # default gmm calibration
    ncc = ncc_search(scene, exemplar, NCCConfig())

    dino_recall = _recall(dino.matches, ground_truth)
    ncc_recall = _recall(ncc.matches, ground_truth)

    assert dino_recall > ncc_recall, f"dino {dino_recall} !> ncc {ncc_recall}"
    # A concrete instance dino-dense found and ncc did not (they truly disagree).
    assert any(_covered(dino.matches, gt) and not _covered(ncc.matches, gt) for gt in ground_truth)


@_needs_model
def test_resolution_cap_engages_and_is_recorded() -> None:
    """A scene above scene_max_side is downscaled and the cap is recorded in diagnostics."""
    scene, exemplar, _ = _pose_varied_scene()  # 840px long side
    result = search(scene, exemplar, DinoDenseConfig(scene_max_side=280))
    assert result.diagnostics.metrics["cap_engaged"] == 1.0
    assert result.diagnostics.metrics["cap_scale"] < 1.0


@_needs_model
def test_self_similarity_calibration_can_return_empty_honestly() -> None:
    """A strict calibration that clears nothing yields outcome=empty with a note, not a crash."""
    scene, exemplar, _ = _pose_varied_scene()
    result = search(scene, exemplar, DinoDenseConfig(calibration="self-similarity"))
    # self-similarity's 0.7 cut is strict for a mean-pooled prototype; either outcome is honest.
    assert result.outcome in (SearchOutcome.OK, SearchOutcome.EMPTY)
    if result.outcome is SearchOutcome.EMPTY:
        assert result.matches == ()
        assert result.diagnostics.notes
