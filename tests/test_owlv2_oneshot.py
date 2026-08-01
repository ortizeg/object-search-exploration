"""Tests for Method 4 (``owlv2-oneshot``) -- OWLv2 image-conditioned one-shot detection.

Two tiers, deliberately (mirroring ``test_propose_retrieve.py``):

* **Model-free logic** -- the config schema, the pure helpers (``_l2_normalize``,
  ``_iou_with_unit_box``, ``select_query_embedding``, ``boxes_to_pixels``, the OWLv2 preprocessing
  tensor), the full ``search`` path driven by an **injected stub inferencer** (so the compose /
  threshold / NMS / exemplar-labelling logic is gated with no weight), the "runtime imports no
  torch" constraint, and the model-absent error path. These **run in CI**.
* **Real-model behaviour** -- the end-to-end search on the real ONNX graph. Needs the gitignored
  ``owlv2_base_patch16.onnx`` and is **skipped when it is absent**.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from pydantic import ValidationError

from object_search.inference import models
from object_search.inference.owlv2 import (
    OWLV2_IMAGE_SIZE,
    Owlv2Embeddings,
    owlv2_preprocess_tensor,
)
from object_search.schemas import BBox, ExemplarBox, SearchOutcome
from object_search.search import has_method, owlv2_oneshot
from object_search.search.ncc import NCCConfig
from object_search.search.owlv2_oneshot import (
    Owlv2OneshotConfig,
    _iou_with_unit_box,
    _l2_normalize,
    boxes_to_pixels,
    reset_inferencer_cache,
    search,
    select_query_embedding,
)

_MODEL_PATH: Path = models.models_dir() / models.MODEL_REGISTRY["owlv2-base-patch16"].dest
_needs_model = pytest.mark.skipif(
    not _MODEL_PATH.is_file(),
    reason=f"owlv2-oneshot needs {_MODEL_PATH} (gitignored; run pixi run -e export export-owlv2)",
)


@pytest.fixture(autouse=True)
def _isolate_cache() -> object:
    """Reset the module-level singleton around every test so monkeypatching cannot leak."""
    reset_inferencer_cache()
    yield
    reset_inferencer_cache()


def _embeddings(embeds: list[list[float]], boxes: list[list[float]]) -> Owlv2Embeddings:
    """Build an Owlv2Embeddings from plain lists (the shape the inferencer returns)."""
    return Owlv2Embeddings(
        class_embeds=np.asarray(embeds, dtype=np.float32),
        boxes_cxcywh=np.asarray(boxes, dtype=np.float32),
    )


class _StubInferencer:
    """A minimal OWLv2 stand-in: returns a fixed query encode for the crop, scene encode otherwise.

    Proves ``search`` needs only something with an ``embed_image`` method returning
    :class:`Owlv2Embeddings` -- the two-image image-guided logic works with no gitignored weight.
    The crop is smaller than the scene, so image size disambiguates which encode to return.
    """

    def __init__(self, query: Owlv2Embeddings, scene: Owlv2Embeddings) -> None:
        self.query = query
        self.scene = scene
        self.calls: list[tuple[int, int]] = []

    def embed_image(self, image: npt.NDArray[np.uint8]) -> Owlv2Embeddings:
        h, w = int(image.shape[0]), int(image.shape[1])
        self.calls.append((h, w))
        return self.query if max(h, w) <= 50 else self.scene


def _scene_stub() -> Owlv2Embeddings:
    """A scene encode: 4 patches matching the query (box[0] overlaps the exemplar), 4 that don't."""
    high = [1.0, 0.0, 0.0, 0.0]
    low = [0.0, 1.0, 0.0, 0.0]
    embeds = [high, high, high, high, low, low, low, low]
    boxes = [
        [0.125, 0.125, 0.15, 0.15],  # -> pixels (10,10,30,30), overlaps the exemplar
        [0.4, 0.125, 0.1, 0.1],
        [0.6, 0.125, 0.1, 0.1],
        [0.8, 0.125, 0.1, 0.1],
        [0.2, 0.6, 0.1, 0.1],
        [0.4, 0.6, 0.1, 0.1],
        [0.6, 0.6, 0.1, 0.1],
        [0.8, 0.6, 0.1, 0.1],
    ]
    return _embeddings(embeds, boxes)


def _query_stub() -> Owlv2Embeddings:
    """A query encode: one patch whose box covers the whole crop, embedding [1,0,0,0]."""
    return _embeddings([[1.0, 0.0, 0.0, 0.0]], [[0.5, 0.5, 1.0, 1.0]])


# ------------------------------------------------------------------- model-free: the config


def test_config_defaults() -> None:
    cfg = Owlv2OneshotConfig()
    assert cfg.score_threshold is None
    assert cfg.calibration == "self-similarity"  # gmm degenerates on OWLv2's compressed cosine
    assert cfg.retain_frac == 0.94  # the robust sweet spot from the retain_frac sweep
    assert cfg.query_iou_frac == 0.8
    assert cfg.max_box_area_frac == 0.25  # drop the generic whole-frame box
    assert cfg.nms_iou == 0.3  # tight NMS collapses OWLv2's per-object duplicate patches
    assert cfg.max_candidates == 50
    assert cfg.seed == 0


def test_config_is_frozen_and_schema_drives_the_form() -> None:
    cfg = Owlv2OneshotConfig()
    with pytest.raises(ValidationError):  # frozen -> mutation is an error
        cfg.nms_iou = 0.9  # type: ignore[misc]
    schema = Owlv2OneshotConfig.model_json_schema()
    for field in (
        "score_threshold",
        "calibration",
        "retain_frac",
        "query_iou_frac",
        "max_box_area_frac",
        "nms_iou",
        "max_candidates",
    ):
        assert schema["properties"][field].get("description")


def test_registered_under_owlv2_oneshot_with_its_config() -> None:
    assert has_method("owlv2-oneshot")
    from object_search.search import get_method

    spec = get_method("owlv2-oneshot")
    assert spec.config_model is Owlv2OneshotConfig


# ------------------------------------------------- model-free: the runtime imports no torch


def test_runtime_modules_import_no_torch_or_transformers() -> None:
    """ "ONNX Runtime for every learned model": the runtime path never imports torch/transformers.

    Scans the ``import`` statements of both runtime modules only -- the docstrings legitimately
    *mention* torch/transformers (they explain the export env), so a naive substring search would
    flag their own prose.
    """
    from object_search.inference import owlv2 as owlv2_inf

    for module in (owlv2_inf, owlv2_oneshot):
        source = Path(module.__file__).read_text(encoding="utf-8")  # type: ignore[arg-type]
        import_lines = [
            line for line in source.splitlines() if line.strip().startswith(("import ", "from "))
        ]
        for banned in ("torch", "transformers", "ultralytics"):
            assert not any(banned in line for line in import_lines), (
                f"{module.__name__} must not import {banned}"
            )


# ------------------------------------------------- model-free: the pure helpers


def test_l2_normalize_makes_unit_rows_and_keeps_zero_zero() -> None:
    vecs = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
    out = _l2_normalize(vecs, axis=1)
    assert np.linalg.norm(out[0]) == pytest.approx(1.0)
    assert np.linalg.norm(out[1]) == pytest.approx(0.0)  # zero stays zero, never NaN
    assert np.isfinite(out).all()


def test_iou_with_unit_box() -> None:
    boxes = np.array(
        [
            [0.5, 0.5, 1.0, 1.0],  # exactly the unit box -> IoU 1.0
            [0.25, 0.25, 0.5, 0.5],  # a quarter, fully inside -> 0.25 / 1.0
        ],
        dtype=np.float32,
    )
    ious = _iou_with_unit_box(boxes)
    assert ious[0] == pytest.approx(1.0, abs=1e-5)
    assert ious[1] == pytest.approx(0.25, abs=1e-5)


def test_select_query_embedding_picks_the_most_distinctive_covering_patch() -> None:
    """Among the covering patches, the one LEAST similar to the mean (the object) is chosen.

    This is the correctness-critical HF heuristic: mean-pooling the covering patches instead would
    return the generic whole-frame direction, which matches background boxes in the scene.
    """
    # Patches 0,1 cover the crop; 2,3 are tiny fillers that pull the mean toward [1,0,0]. Patch 1
    # is the distinctive (orthogonal) covering patch -- the object -- and must be the one selected.
    class_embeds = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32
    )
    boxes = np.array(
        [
            [0.5, 0.5, 1.0, 1.0],  # covering, generic (aligned with the mean)
            [0.5, 0.5, 1.0, 1.0],  # covering, distinctive (orthogonal to the mean)
            [0.05, 0.05, 0.02, 0.02],  # tiny filler
            [0.05, 0.05, 0.02, 0.02],  # tiny filler
        ],
        dtype=np.float32,
    )
    q = select_query_embedding(class_embeds, boxes, iou_frac=0.8)
    assert q.shape == (3,)
    assert np.linalg.norm(q) == pytest.approx(1.0, abs=1e-5)
    assert q[1] > q[0]  # the distinctive [0,1,0] patch, NOT the generic [1,0,0] one


def test_boxes_to_pixels_maps_and_drops_degenerate() -> None:
    boxes = np.array(
        [
            [0.125, 0.125, 0.15, 0.15],  # -> (10,10,30,30) at side 200
            [0.0, 0.0, 0.0, 0.0],  # degenerate -> None
        ],
        dtype=np.float32,
    )
    out = boxes_to_pixels(boxes, orig_w=200, orig_h=200)
    assert out[0] == BBox(x=10, y=10, w=30, h=30)
    assert out[1] is None


def test_owlv2_preprocess_tensor_shape_and_side() -> None:
    """The preprocessing produces the fixed [1,3,960,960] tensor and reports the square side."""
    image = np.zeros((80, 120, 3), dtype=np.uint8)  # non-square -> side is max(H, W) = 120
    tensor, side = owlv2_preprocess_tensor(image)
    assert tensor.shape == (1, 3, OWLV2_IMAGE_SIZE, OWLV2_IMAGE_SIZE)
    assert tensor.dtype == np.float32
    assert side == 120


# ------------------------------------------------- model-free: the full search path (stubbed)


def test_search_end_to_end_with_a_stub_inferencer(monkeypatch: pytest.MonkeyPatch) -> None:
    """search composes query + scene encodes, thresholds, NMS, labels the exemplar -- no weight."""
    stub = _StubInferencer(_query_stub(), _scene_stub())
    monkeypatch.setattr(owlv2_oneshot, "_get_inferencer", lambda: stub)

    scene = np.zeros((200, 200, 3), dtype=np.uint8)
    exemplar = ExemplarBox(box=BBox(x=10, y=10, w=30, h=30))
    # Fixed threshold => deterministic (no gmm dependency): the four high-cosine patches clear 0.5.
    result = search(scene, exemplar, Owlv2OneshotConfig(score_threshold=0.5))

    assert result.outcome is SearchOutcome.OK
    assert len(result.matches) == 4  # METHOD-12: every accepted box, not a single-best
    assert sum(m.is_exemplar for m in result.matches) == 1  # the box overlapping the exemplar
    assert all(m.transform is None for m in result.matches)  # appearance detector, no affine
    # Two forward passes: the crop encode and the scene encode, attributed separately (EVAL-11).
    assert stub.calls[0] == (30, 30) and stub.calls[1] == (200, 200)
    assert "query_ms" in result.diagnostics.metrics
    assert "target_ms" in result.diagnostics.metrics
    assert result.diagnostics.proposals is not None


def test_search_empty_when_nothing_clears(monkeypatch: pytest.MonkeyPatch) -> None:
    """When no patch clears the threshold, the result is EMPTY-with-note, never a raise."""
    stub = _StubInferencer(_query_stub(), _scene_stub())
    monkeypatch.setattr(owlv2_oneshot, "_get_inferencer", lambda: stub)

    scene = np.zeros((200, 200, 3), dtype=np.uint8)
    exemplar = ExemplarBox(box=BBox(x=10, y=10, w=30, h=30))
    # Acceptance is strict (score > threshold); threshold 1.0 rejects even exact-1.0 cosines.
    result = search(scene, exemplar, Owlv2OneshotConfig(score_threshold=1.0))

    assert result.outcome is SearchOutcome.EMPTY
    assert result.matches == ()
    assert result.diagnostics.notes  # a note explains nothing cleared the threshold


def test_search_drops_the_generic_whole_frame_box(monkeypatch: pytest.MonkeyPatch) -> None:
    """A predicted box larger than max_box_area_frac of the image is never returned or ranked."""
    query = _query_stub()  # query embedding is [1, 0, 0, 0]
    # scene 200x200 (area 40000), cap = 0.25 * 40000 = 10000. The whole-frame box scores highest
    # (cosine 1.0) but must be filtered out so it neither matches nor dominates NMS.
    scene = _embeddings(
        [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        [
            [0.5, 0.5, 1.0, 1.0],  # whole frame -> (0,0,200,200) area 40000 > cap: DROPPED
            [0.1, 0.1, 0.15, 0.15],  # object -> (5,5,30,30) area 900: kept
            [0.7, 0.7, 0.1, 0.1],  # background, low score
        ],
    )
    stub = _StubInferencer(query, scene)
    monkeypatch.setattr(owlv2_oneshot, "_get_inferencer", lambda: stub)

    exemplar = ExemplarBox(box=BBox(x=5, y=5, w=30, h=30))
    result = search(
        np.zeros((200, 200, 3), dtype=np.uint8), exemplar, Owlv2OneshotConfig(score_threshold=0.5)
    )
    assert result.outcome is SearchOutcome.OK
    cap = 0.25 * 200 * 200
    assert all(m.box.area <= cap for m in result.matches)  # no whole-frame box survived
    assert all(c.box.area <= cap for c in result.candidates)


# ------------------------------------------------- model-free: the model-absent error path


def test_search_returns_model_unavailable_when_weight_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no weight on disk, search degrades to outcome=error, never a raise."""
    monkeypatch.setattr(owlv2_oneshot.models, "models_dir", lambda: tmp_path)
    reset_inferencer_cache()
    scene = np.full((60, 60, 3), 50, dtype=np.uint8)
    result = search(scene, ExemplarBox(box=BBox(x=5, y=5, w=20, h=20)), Owlv2OneshotConfig())
    assert result.outcome is SearchOutcome.ERROR
    assert result.error is not None
    assert result.error.kind == "model_unavailable"
    assert result.matches == ()


# --------------------------------------- model-free: which weight file the method resolves (8zy)


def test_resolve_model_path_without_the_env_var_is_the_shipped_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unset OS_OWLV2_MODEL => exactly the path this method resolved before the override existed.

    This is the whole promise of the opt-in: with no env var set, the shipped pretrained graph is
    what loads, so every previously reported owlv2-oneshot number stays comparable.
    """
    monkeypatch.delenv("OS_OWLV2_MODEL", raising=False)
    expected = models.models_dir() / models.MODEL_REGISTRY["owlv2-base-patch16"].dest
    assert owlv2_oneshot._resolve_model_path() == expected

    # An empty / whitespace value is treated as unset, not as a path to "".
    monkeypatch.setenv("OS_OWLV2_MODEL", "   ")
    assert owlv2_oneshot._resolve_model_path() == expected


def test_resolve_model_path_relative_lands_in_models_dir_absolute_is_verbatim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(owlv2_oneshot.models, "models_dir", lambda: tmp_path)

    monkeypatch.setenv("OS_OWLV2_MODEL", "owlv2_base_patch16_floorplans_ft.onnx")
    assert owlv2_oneshot._resolve_model_path() == (
        tmp_path / "owlv2_base_patch16_floorplans_ft.onnx"
    )

    absolute = tmp_path / "elsewhere" / "arm_b.onnx"
    monkeypatch.setenv("OS_OWLV2_MODEL", str(absolute))
    assert owlv2_oneshot._resolve_model_path() == absolute


def test_method_version_and_config_defaults_are_unchanged_by_the_weight_override() -> None:
    """The override points the SAME method at a different file; it is not a new method (8zy).

    If `_METHOD_VERSION` or any config default moved, an A/B across weights would no longer be
    attributable to the weights alone -- which is the only reason the override exists.
    """
    assert owlv2_oneshot._METHOD_VERSION == "1.0.0"
    config = Owlv2OneshotConfig()
    assert config.calibration == "self-similarity"
    assert config.retain_frac == 0.94
    assert config.query_iou_frac == 0.8
    assert config.max_box_area_frac == 0.25
    assert config.nms_iou == 0.3
    assert config.max_candidates == 50
    assert config.score_threshold is None
    assert config.seed == 0


def test_search_rejects_a_foreign_config() -> None:
    scene = np.full((60, 60, 3), 50, dtype=np.uint8)
    with pytest.raises(TypeError, match="requires an Owlv2OneshotConfig"):
        search(scene, ExemplarBox(box=BBox(x=5, y=5, w=20, h=20)), NCCConfig())


# ================================================== real-model behaviour (skipped in CI)


@_needs_model
def test_end_to_end_finds_instances_on_the_real_graph() -> None:
    """search runs the real OWLv2 ONNX graph end to end and returns a well-formed result."""
    from object_search.synthetic.chipset import CHIPSET_SPECS, generate_chipset_image

    bundle = generate_chipset_image(CHIPSET_SPECS[2])  # 800x600
    exemplar = ExemplarBox(box=bundle.boxes[0])
    result = search(bundle.image, exemplar, Owlv2OneshotConfig())
    assert result.outcome in (SearchOutcome.OK, SearchOutcome.EMPTY)
    assert "target_ms" in result.diagnostics.metrics
