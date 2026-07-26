"""Tests for Method 5 (``propose-retrieve``) -- the two-unit compose method.

Two tiers, deliberately (mirroring ``test_dino_dense.py`` / ``test_proposals.py``):

* **Model-free logic** -- the config schema, the L2-normalization, the ``embed_regions`` unit with
  an injected stub inferencer, the **seam** (``propose`` and ``embed_regions`` each called directly,
  never through ``search``), the **reuse** of Method 3's DINOv2 singleton (proving no second model
  is loaded/fetched), the no-FAISS guard, and the model-absent error path. These need no weight and
  **run in CI**, gating the risky compose/threshold/NMS logic.
* **Real-model behaviour** -- the end-to-end search, the latency split, and the headline success
  criterion (**boundary-alignment IoU** against the chipset ground truth). These need the gitignored
  ``fastsam_s.onnx`` and ``dinov2_small.onnx`` and are **skipped when either is absent**.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from pydantic import ValidationError

from object_search.inference import models
from object_search.schemas import BBox, ExemplarBox, SearchOutcome
from object_search.search import dino_dense, has_method, propose_retrieve
from object_search.search.ncc import NCCConfig
from object_search.search.propose_retrieve import (
    ProposeRetrieveConfig,
    _l2_normalize,
    embed_regions,
    reset_backend_cache,
    search,
)
from object_search.synthetic.chipset import CHIPSET_SPECS, generate_chipset_image

_DIM = 384

_FASTSAM_PATH: Path = models.models_dir() / models.MODEL_REGISTRY["fastsam-s"].dest
_DINOV2_PATH: Path = models.models_dir() / models.MODEL_REGISTRY["dinov2-small"].dest
_HAVE_MODELS: bool = _FASTSAM_PATH.is_file() and _DINOV2_PATH.is_file()
_needs_models = pytest.mark.skipif(
    not _HAVE_MODELS,
    reason=(
        f"propose-retrieve needs both weights: fastsam-s at {_FASTSAM_PATH} and dinov2-small at "
        f"{_DINOV2_PATH} (gitignored; run pixi run fetch-models)"
    ),
)


@pytest.fixture(autouse=True)
def _isolate_caches() -> object:
    """Reset both module-level singletons around every test so monkeypatching cannot leak."""
    dino_dense.reset_inferencer_cache()
    reset_backend_cache()
    yield
    dino_dense.reset_inferencer_cache()
    reset_backend_cache()


class _StubInferencer:
    """A minimal DINOv2 stand-in: records its calls and returns a fixed token grid.

    Proves ``embed_regions`` is an independently callable unit -- it needs only something with a
    ``dense_tokens`` method, so the seam works with no gitignored weight.
    """

    def __init__(self, fill: float = 1.0) -> None:
        self.calls: list[tuple[int, ...]] = []
        self.fill = fill

    def dense_tokens(
        self, image: npt.NDArray[np.uint8]
    ) -> tuple[npt.NDArray[np.float32], float, float]:
        self.calls.append(tuple(int(s) for s in image.shape))
        grid = np.full((2, 3, _DIM), self.fill, dtype=np.float32)
        return grid, 1.0, 1.0


# ------------------------------------------------------------------- model-free: the config


def test_config_defaults_match_the_locked_decisions() -> None:
    cfg = ProposeRetrieveConfig()
    assert cfg.proposal_backend == "fastsam"
    assert cfg.proposal_conf == 0.4
    assert cfg.retrieval_threshold is None
    assert cfg.nms_iou == 0.3
    assert cfg.similarity_floor == 0.7
    assert cfg.max_candidates == 50
    assert cfg.seed == 0


def test_config_is_frozen_and_schema_drives_the_form() -> None:
    cfg = ProposeRetrieveConfig()
    with pytest.raises(ValidationError):  # frozen -> mutation is an error
        cfg.nms_iou = 0.9  # type: ignore[misc]
    schema = ProposeRetrieveConfig.model_json_schema()
    for field in (
        "proposal_backend",
        "proposal_conf",
        "retrieval_threshold",
        "nms_iou",
        "similarity_floor",
    ):
        assert schema["properties"][field].get("description")


def test_registered_under_propose_retrieve_with_its_config() -> None:
    assert has_method("propose-retrieve")
    from object_search.search import get_method

    spec = get_method("propose-retrieve")
    assert spec.config_model is ProposeRetrieveConfig


# ---------------------------------------------------- model-free: no FAISS anywhere (a constraint)


def test_module_imports_no_faiss() -> None:
    """FAISS is deliberately NOT adopted in Milestone 1 (CONTEXT decision 7): plain NumPy matmul.

    Scans the module's ``import`` statements only -- the docstring legitimately *explains* the
    no-FAISS decision, so a naive substring search would flag its own rationale.
    """
    source = Path(propose_retrieve.__file__).read_text(encoding="utf-8")
    import_lines = [
        line for line in source.splitlines() if line.strip().startswith(("import ", "from "))
    ]
    assert not any("faiss" in line.lower() for line in import_lines), (
        "propose-retrieve must not import FAISS"
    )


# ---------------------------------------------- model-free: L2-normalization (a load-bearing truth)


def test_l2_normalize_makes_unit_rows_and_keeps_zero_zero() -> None:
    vecs = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
    out = _l2_normalize(vecs, axis=1)
    assert np.linalg.norm(out[0]) == pytest.approx(1.0)
    assert np.linalg.norm(out[1]) == pytest.approx(0.0)  # zero stays zero, never NaN
    assert np.isfinite(out).all()


# ------------------------------------ model-free: embed_regions is an independently callable unit


def test_embed_regions_returns_one_l2_normalized_row_per_box() -> None:
    """Call embed_regions DIRECTLY (the seam), with an injected stub inferencer -- no weight."""
    stub = _StubInferencer(fill=2.0)
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    boxes = [BBox(x=0, y=0, w=20, h=20), BBox(x=30, y=30, w=25, h=25)]
    embeddings = embed_regions(image, boxes, ProposeRetrieveConfig(), inferencer=stub)

    assert embeddings.shape == (2, _DIM)
    assert embeddings.dtype == np.float32
    # Each row is a unit vector (mean-pool then L2-normalize).
    assert np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-5)
    # It embedded exactly the two boxes, cropping each from the scene.
    assert len(stub.calls) == 2


def test_embed_regions_on_empty_boxes_returns_empty_matrix() -> None:
    stub = _StubInferencer()
    out = embed_regions(
        np.zeros((10, 10, 3), np.uint8), [], ProposeRetrieveConfig(), inferencer=stub
    )
    assert out.shape == (0, _DIM)


def test_embed_regions_upsizes_a_subpatch_crop() -> None:
    """A proposal smaller than one 14px patch is up-sized so it still yields a token (no crash)."""
    stub = _StubInferencer()
    image = np.zeros((40, 40, 3), dtype=np.uint8)
    out = embed_regions(image, [BBox(x=0, y=0, w=5, h=5)], ProposeRetrieveConfig(), inferencer=stub)
    assert out.shape == (1, _DIM)
    # The crop handed to the backbone was padded up to at least 14x14.
    assert stub.calls[0][0] >= 14 and stub.calls[0][1] >= 14


def test_embed_regions_rejects_a_foreign_config() -> None:
    stub = _StubInferencer()
    with pytest.raises(TypeError, match="ProposeRetrieveConfig"):
        embed_regions(
            np.zeros((10, 10, 3), np.uint8),
            [BBox(x=0, y=0, w=4, h=4)],
            NCCConfig(),
            inferencer=stub,
        )


def test_embed_regions_without_a_backbone_raises_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    """No injected inferencer and the shared weight absent => a loud RuntimeError, not empty."""
    monkeypatch.setattr(dino_dense, "_get_inferencer", lambda: None)
    with pytest.raises(RuntimeError, match="dinov2-small"):
        embed_regions(
            np.zeros((10, 10, 3), np.uint8), [BBox(x=0, y=0, w=4, h=4)], ProposeRetrieveConfig()
        )


# ----------------------------- model-free: the REUSE contract (one DINOv2, shared with Method 3)


def test_embed_regions_reuses_the_dino_dense_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no injected inferencer, embed_regions routes through Method 3's DINOv2 singleton.

    This is the "no second DINOv2 model is fetched" guarantee (Phase 7 success criterion 3): the
    embedding stage does not construct or fetch its own backbone -- it uses the ONE that Method 3
    already loaded. A spy standing in for that singleton proves the delegation.
    """
    spy = _StubInferencer()
    monkeypatch.setattr(dino_dense, "_get_inferencer", lambda: spy)
    out = embed_regions(
        np.zeros((32, 32, 3), np.uint8), [BBox(x=0, y=0, w=16, h=16)], ProposeRetrieveConfig()
    )
    assert out.shape == (1, _DIM)
    assert len(spy.calls) == 1  # it used the shared singleton, not a fresh model


def test_there_is_exactly_one_dinov2_in_the_model_registry() -> None:
    """No second DINOv2 model key exists to fetch -- the reuse is structural, not incidental."""
    dinov2_keys = [k for k in models.MODEL_REGISTRY if "dinov2" in k.lower()]
    assert dinov2_keys == ["dinov2-small"]


# ---------------------------------------------------- model-free: the model-absent error path


def test_search_returns_model_unavailable_when_weights_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no weights on disk, search degrades to outcome=error, never a raise."""
    monkeypatch.setattr(propose_retrieve.models, "models_dir", lambda: tmp_path)
    monkeypatch.setattr(dino_dense.models, "models_dir", lambda: tmp_path)
    dino_dense.reset_inferencer_cache()
    reset_backend_cache()
    scene = np.full((60, 60, 3), 50, dtype=np.uint8)
    result = search(scene, ExemplarBox(box=BBox(x=5, y=5, w=20, h=20)), ProposeRetrieveConfig())
    assert result.outcome is SearchOutcome.ERROR
    assert result.error is not None
    assert result.error.kind == "model_unavailable"
    assert result.matches == ()


def test_search_rejects_a_foreign_config() -> None:
    scene = np.full((60, 60, 3), 50, dtype=np.uint8)
    with pytest.raises(TypeError, match="requires a ProposeRetrieveConfig"):
        search(scene, ExemplarBox(box=BBox(x=5, y=5, w=20, h=20)), NCCConfig())


# --------------- model-free: the degenerate single-mode floor (the uniform-lattice regression)


class _FixedProposalBackend:
    """A stub proposal backend returning a fixed set of boxes -- no FastSAM weight needed."""

    def __init__(self, boxes: list[BBox]) -> None:
        self._boxes = boxes

    def propose(self, image: npt.NDArray[np.uint8], config: object) -> list:
        from object_search.inference import Proposal

        return [Proposal(box=b, mask=None, objectness=0.9) for b in self._boxes]


def test_uniform_lattice_is_not_rejected_by_the_similarity_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A uniform lattice of identical instances -- all cosines ~1.0, a single mode -- must NOT be
    rejected. Regression guard for the calibration catastrophe: before the ``similarity_floor``,
    the gmm degeneracy fallback cut *at* the max score and the strict ``> threshold`` rejected
    every true match (recall 0), the worst failure for a repeated-instance finder.
    """
    # Four non-overlapping identical instances (so NMS keeps them all) plus the exemplar's own box.
    boxes = [BBox(x=x, y=y, w=20, h=20) for y in (5, 55) for x in (5, 55)]
    scene = np.full((120, 120, 3), 50, dtype=np.uint8)

    # Stub the two singletons: identical embeddings for every crop => every cosine is exactly 1.0,
    # so np.unique(scores).size < 2 and the gmm is degenerate -- the exact lattice-touching case.
    monkeypatch.setattr(propose_retrieve, "_get_backend", lambda: _FixedProposalBackend(boxes))
    monkeypatch.setattr(dino_dense, "_get_inferencer", lambda: _StubInferencer(fill=1.0))

    result = search(scene, ExemplarBox(box=boxes[0]), ProposeRetrieveConfig())

    assert result.outcome is SearchOutcome.OK
    assert len(result.matches) == len(boxes)  # every identical instance is found, not zero
    assert result.diagnostics.metrics["threshold"] == pytest.approx(0.7)  # the floor decided


# ================================================== real-model behaviour (skipped in CI)


def _mean_iou_to_ground_truth(matches: tuple, ground_truth: list[BBox]) -> float:
    """Mean over GT boxes of the best IoU with any returned match box -- the alignment number."""
    total = 0.0
    for gt in ground_truth:
        total += max((gt.iou(m.box) for m in matches), default=0.0)
    return total / len(ground_truth)


@_needs_models
def test_end_to_end_finds_instances_with_diagnostics_and_latency_split() -> None:
    """search composes propose + embed, returns matches, and reports the two stages separately."""
    result_bundle = generate_chipset_image(CHIPSET_SPECS[4])  # 1600x1200, ~72px chips
    scene = result_bundle.image
    ground_truth = list(result_bundle.boxes)
    exemplar = ExemplarBox(box=ground_truth[0])

    result = search(scene, exemplar, ProposeRetrieveConfig())

    assert result.outcome is SearchOutcome.OK
    assert len(result.matches) > 1  # METHOD-12: many instances, not a single-best
    # The proposal set is carried for the UI diagnostics overlay (Phase 7 criterion 4).
    assert result.diagnostics.proposals is not None
    assert len(result.diagnostics.proposals) >= len(result.matches)
    # Latency attributes proposal vs embedding time SEPARATELY (EVAL-11).
    assert "proposal_ms" in result.diagnostics.metrics
    assert "embedding_ms" in result.diagnostics.metrics
    assert "n_proposals" in result.diagnostics.metrics
    # Exactly one match is the exemplar's own region.
    assert sum(m.is_exemplar for m in result.matches) <= 1


@_needs_models
def test_boxes_hug_object_boundaries_by_iou_against_ground_truth() -> None:
    """Phase 7 success criterion 1 as a NUMBER: mean IoU vs exact chipset GT is high.

    This is the method's selling point over dino-dense's blobby components -- FastSAM proposals
    hug the chip boundaries, so the returned boxes align tightly to the exact ground truth.
    """
    bundle = generate_chipset_image(CHIPSET_SPECS[4])
    scene = bundle.image
    ground_truth = list(bundle.boxes)
    exemplar = ExemplarBox(box=ground_truth[0])

    result = search(scene, exemplar, ProposeRetrieveConfig())
    assert result.outcome is SearchOutcome.OK

    mean_iou = _mean_iou_to_ground_truth(result.matches, ground_truth)
    assert mean_iou > 0.7, f"boundary alignment too loose: mean IoU {mean_iou:.3f} !> 0.70"


@_needs_models
def test_embed_regions_real_backbone_standalone_is_normalized() -> None:
    """The embedding unit works directly on the real backbone (the seam, model side)."""
    bundle = generate_chipset_image(CHIPSET_SPECS[2])  # 800x600
    boxes = list(bundle.boxes[:3])
    embeddings = embed_regions(bundle.image, boxes, ProposeRetrieveConfig())
    assert embeddings.shape == (len(boxes), _DIM)
    assert np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-4)
