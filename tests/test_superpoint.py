"""Tests for the SuperPoint learned backend of Method 2 (`sparse-geo`).

Two tiers, deliberately (mirroring `test_dinov2.py`):

* **Model-free** -- the frozen grayscale input spec, the docstring contract, the frameless
  keypoint mapping, and the config-only voting-mode rules (single-4dof rejected for superpoint,
  translation-2dof the superpoint default). These need no weight and so **run in CI**, which is
  where the load-bearing "single-4dof + superpoint raises" rule is gated.
* **Real-model** -- loading `superpoint.onnx` and asserting the verified I/O contract
  (int64 keypoints, L2-normalized descriptors, 8-px border, variable count). These need the
  gitignored weight and are **skipped when it is absent**, exactly as the phase context requires.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from object_search.inference import models
from object_search.inference.superpoint import (
    SUPERPOINT_BORDER_PX,
    SUPERPOINT_DESCRIPTOR_DIM,
    SUPERPOINT_INPUT_SPEC,
    SUPERPOINT_STRIDE,
    SuperPointInferencer,
    SuperPointResult,
)
from object_search.search.sparse_geo import (
    SparseGeoConfig,
    _detect_superpoint,
)

# Pin the CPU provider so real-model runs are identical run to run (the dev machine also
# exposes CoreML).
_CPU = ["CPUExecutionProvider"]

_MODEL_PATH: Path = models.models_dir() / models.MODEL_REGISTRY["superpoint"].dest
_HAVE_MODEL: bool = _MODEL_PATH.is_file()
_needs_model = pytest.mark.skipif(
    not _HAVE_MODEL,
    reason=f"superpoint weight absent at {_MODEL_PATH} (gitignored; run pixi run fetch-models)",
)


# ------------------------------------------------------- model-free: the input-spec contract


def test_input_spec_is_grayscale_single_channel_no_meanstd() -> None:
    """The frozen spec IS the preprocessing contract; pin its verified fields."""
    spec = SUPERPOINT_INPUT_SPEC
    assert spec.input_name == "image"
    assert spec.dtype == "tensor(float)"
    # [1, 1, H, W] -- batch fixed at 1, ONE grayscale channel, H/W dynamic (str dims).
    assert spec.shape[0] == 1
    assert spec.shape[1] == 1, "SuperPoint takes a single grayscale channel"
    assert isinstance(spec.shape[2], str) and isinstance(spec.shape[3], str)
    assert spec.layout == "NCHW"
    # Range [0, 1] via 1/255, and NO mean/std normalization (identity mean 0 / std 1).
    assert spec.scale == pytest.approx(1.0 / 255.0)
    assert spec.mean == (0.0, 0.0, 0.0)
    assert spec.std == (1.0, 1.0, 1.0)


def test_docstring_states_the_verified_contract() -> None:
    """CLAUDE.md requires the exact pre/post numbers in the inferencer docstring."""
    import object_search.inference.superpoint as module

    doc = module.__doc__ or ""
    for needle in (
        "grayscale",
        "1/255",
        "NO mean subtraction",
        "L2-normalized",
        "int64",
        "8 px",
        "non-commercial",
    ):
        assert needle in doc, f"docstring missing {needle!r}"


def test_border_and_stride_constants_are_the_verified_values() -> None:
    # Effective border is 8, NOT the configured remove_borders=4; stride is 8; descriptors 256-D.
    assert SUPERPOINT_BORDER_PX == 8
    assert SUPERPOINT_STRIDE == 8
    assert SUPERPOINT_DESCRIPTOR_DIM == 256


# --------------------------------------------- model-free: the backend keypoints are frameless


class _StubInferencer:
    """A duck-typed stand-in for SuperPointInferencer.detect, so the frameless mapping is testable
    without the gitignored weight."""

    def __init__(self, result: SuperPointResult) -> None:
        self._result = result

    def detect(self, image: npt.NDArray[np.uint8]) -> SuperPointResult:
        return self._result


def test_superpoint_backend_keypoints_carry_no_scale_or_angle() -> None:
    """The load-bearing property: SuperPoint keypoints have no frame, so scale/angle are None."""
    result = SuperPointResult(
        keypoints=np.array([[10, 20], [30, 40]], dtype=np.int64),
        scores=np.array([0.9, 0.8], dtype=np.float32),
        descriptors=np.zeros((2, SUPERPOINT_DESCRIPTOR_DIM), dtype=np.float32),
    )
    gray = np.zeros((64, 64), dtype=np.uint8)
    kps = _detect_superpoint(gray, _StubInferencer(result), origin_xy=(5, 7))  # type: ignore[arg-type]

    assert kps.scale is None, "a frameless backend must expose no keypoint scale"
    assert kps.angle is None, "a frameless backend must expose no keypoint orientation"
    # Coordinates are shifted by the crop origin into scene pixels.
    assert kps.xy.tolist() == [[15.0, 27.0], [35.0, 47.0]]
    assert kps.count == 2


def test_superpoint_backend_maps_empty_detection_to_empty_keypoints() -> None:
    empty = SuperPointResult(
        keypoints=np.empty((0, 2), dtype=np.int64),
        scores=np.empty((0,), dtype=np.float32),
        descriptors=np.empty((0, SUPERPOINT_DESCRIPTOR_DIM), dtype=np.float32),
    )
    gray = np.zeros((32, 32), dtype=np.uint8)
    kps = _detect_superpoint(gray, _StubInferencer(empty), origin_xy=(0, 0))  # type: ignore[arg-type]
    assert kps.count == 0
    assert kps.scale is None and kps.angle is None


# --------------------------------------------- model-free: voting-MODE rules (config-only)
# (named with "mode" so `pytest -k mode` selects exactly these, per the plan's Task 2 verify.)


def test_superpoint_single_4dof_mode_raises_frameless() -> None:
    """single-4dof + superpoint is refused at config time, not silently degraded (METHOD-04a)."""
    with pytest.raises(ValueError, match="single-4dof"):
        SparseGeoConfig(backend="superpoint", voting_mode="single-4dof")


def test_superpoint_default_voting_mode_is_translation_2dof() -> None:
    """Omitting voting_mode with the superpoint backend yields the translation-2dof default."""
    config = SparseGeoConfig(backend="superpoint")
    assert config.voting_mode == "translation-2dof"


def test_superpoint_translation_2dof_mode_is_accepted() -> None:
    config = SparseGeoConfig(backend="superpoint", voting_mode="translation-2dof")
    assert config.backend == "superpoint"
    assert config.voting_mode == "translation-2dof"


def test_superpoint_pairwise_4dof_mode_is_accepted() -> None:
    config = SparseGeoConfig(backend="superpoint", voting_mode="pairwise-4dof")
    assert config.voting_mode == "pairwise-4dof"


def test_classical_backend_default_voting_mode_is_unchanged_single_4dof() -> None:
    """The superpoint default must not disturb the classical single-4dof default."""
    assert SparseGeoConfig().voting_mode == "single-4dof"
    assert SparseGeoConfig(backend="sift").voting_mode == "single-4dof"


# ------------------------------------------------------------- real-model: the verified contract


@_needs_model
def test_construction_validates_the_grayscale_input_at_load() -> None:
    # A wrong-channel model would fail construction (INFRA-09); the real weight must load clean.
    inf = SuperPointInferencer(_MODEL_PATH, providers=_CPU)
    assert inf.input_spec.shape[1] == 1


@_needs_model
def test_outputs_are_int64_keypoints_and_l2_normalized_descriptors() -> None:
    """int64 (x, y), 256-D descriptors with ||d|| == 1, variable N, all within the 8-px border."""
    rng = np.random.default_rng(0)
    scene = rng.integers(0, 256, size=(240, 320, 3), dtype=np.uint8)
    inf = SuperPointInferencer(_MODEL_PATH, providers=_CPU)
    result = inf.detect(scene)

    assert result.keypoints.dtype == np.int64
    assert result.keypoints.shape[1] == 2
    assert result.descriptors.shape[1] == SUPERPOINT_DESCRIPTOR_DIM
    # Aligned lengths (one shared symbolic N).
    n = result.keypoints.shape[0]
    assert result.scores.shape == (n,)
    assert result.descriptors.shape[0] == n
    assert n > 0, "a textured scene must yield keypoints"

    # Descriptors are ALREADY L2-normalized -- do not re-normalize.
    norms = np.linalg.norm(result.descriptors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3)

    # Effective 8-px border: no keypoint lands within 8 px of the (padded) edge on the near sides.
    assert result.keypoints[:, 0].min() >= SUPERPOINT_BORDER_PX - 1
    assert result.keypoints[:, 1].min() >= SUPERPOINT_BORDER_PX - 1


@_needs_model
def test_variable_keypoint_count_falls_with_less_texture() -> None:
    """The variable-count property METHOD-04c depends on: a tiny flat crop yields far fewer."""
    inf = SuperPointInferencer(_MODEL_PATH, providers=_CPU)
    rng = np.random.default_rng(1)
    textured = rng.integers(0, 256, size=(240, 320, 3), dtype=np.uint8)
    flat = np.full((240, 320, 3), 128, dtype=np.uint8)
    assert inf.detect(textured).keypoints.shape[0] > inf.detect(flat).keypoints.shape[0]


@_needs_model
def test_end_to_end_superpoint_backend_runs_through_the_shared_path() -> None:
    """backend=superpoint + translation-2dof runs through the same search() as classical."""
    import cv2

    from object_search.schemas import BBox, ExemplarBox
    from object_search.search.sparse_geo import search

    rng = np.random.default_rng(3)
    small = rng.integers(0, 256, size=(10, 10), dtype=np.uint8)
    tile = cv2.resize(small, (64, 64), interpolation=cv2.INTER_CUBIC)
    scene = np.full((360, 600), 128, dtype=np.uint8)
    for x, y in [(20, 20), (180, 25), (340, 30), (60, 200), (240, 210), (460, 200)]:
        scene[y : y + 64, x : x + 64] = tile
    bgr = np.ascontiguousarray(np.stack([scene] * 3, axis=-1))
    exemplar = ExemplarBox(box=BBox(x=20, y=20, w=64, h=64))

    result = search(bgr, exemplar, SparseGeoConfig(backend="superpoint"))
    # It must at least run cleanly through the shared code path and produce a diagnostics payload.
    assert result.method == "sparse-geo"
    assert result.diagnostics is not None
