"""Tests for :mod:`object_search.inference.fastsam`.

Two tiers, deliberately (mirroring ``test_dinov2.py``):

* **Model-free decoding arithmetic** -- the YOLOv8-seg decode (transpose, split, confidence
  filter, NMS, mask sigmoid + crop-to-box, letterbox undo) is fed **synthetic** ``output0`` /
  ``output1`` tensors of the verified shapes ``[1, 37, 21504]`` / ``[1, 32, 256, 256]`` and lower
  toy shapes. These need no weight and so **run in CI** to gate the riskiest logic.
* **Real-model behaviour** -- constructing :class:`FastSAMInferencer` and running ``predict`` needs
  the gitignored ``fastsam_s.onnx`` and is **skipped when the weight is absent**, exactly as the
  phase context requires (CI cannot export the AGPL weight).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from object_search.inference import (
    FastSAMConfig,
    FastSAMInferencer,
    Proposal,
    decode_fastsam,
    models,
)
from object_search.inference import (
    fastsam as fastsam_module,
)
from object_search.inference.fastsam import (
    FASTSAM_INPUT_SPEC,
    _crop_masks_to_boxes,
    _letterbox_factors,
)
from object_search.schemas import BBox

_CPU = ["CPUExecutionProvider"]
_MODEL_PATH: Path = models.models_dir() / models.MODEL_REGISTRY["fastsam-s"].dest
_HAVE_MODEL: bool = _MODEL_PATH.is_file()
_needs_model = pytest.mark.skipif(
    not _HAVE_MODEL,
    reason=(
        f"fastsam-s weight absent at {_MODEL_PATH} "
        f"(gitignored AGPL export; run pixi run -e export export-fastsam)"
    ),
)


def _make_output0(
    anchors: int, dets: list[tuple[float, float, float, float, float]]
) -> npt.NDArray[np.float32]:
    """Build a synthetic ``[1, 37, anchors]`` output0 with ``dets`` placed in the first anchors.

    Each det is ``(cx, cy, w, h, conf)`` in letterboxed input pixels; the mask coefficients are
    left at zero unless a test overrides them. All other anchors stay at zero confidence.
    """
    out = np.zeros((1, 37, anchors), dtype=np.float32)
    for a, (cx, cy, w, h, conf) in enumerate(dets):
        out[0, 0, a] = cx
        out[0, 1, a] = cy
        out[0, 2, a] = w
        out[0, 3, a] = h
        out[0, 4, a] = conf
    return out


# --------------------------------------------------------------- model-free: the input contract


def test_input_spec_is_letterbox_1024_rgb_no_meanstd() -> None:
    """The frozen spec is the preprocessing contract; pin its verified fields."""
    spec = FASTSAM_INPUT_SPEC
    assert spec.input_name == "images"
    assert spec.shape == (1, 3, 1024, 1024)
    assert spec.layout == "NCHW"
    assert spec.color_order == "RGB"
    assert spec.resize == "letterbox"
    assert spec.scale == pytest.approx(1.0 / 255.0)
    # YOLO does no mean/std -- the spec carries the inert identities.
    assert spec.mean == (0.0, 0.0, 0.0)
    assert spec.std == (1.0, 1.0, 1.0)


def test_config_defaults_are_fastsam_defaults() -> None:
    cfg = FastSAMConfig()
    assert cfg.conf_thres == pytest.approx(0.4)
    assert cfg.iou_thres == pytest.approx(0.9)
    assert cfg.max_proposals is None
    assert cfg.return_masks is False


def test_docstring_states_the_decoding_and_licence() -> None:
    """CLAUDE.md requires the pre/post steps and the AGPL note written into the docstring."""
    doc = fastsam_module.__doc__ or ""
    for needle in (
        "output0",
        "output1",
        "1/255",
        "letterbox",
        "crop each mask to its own box",
        "AGPL",
    ):
        assert needle in doc, f"docstring missing {needle!r}"


# -------------------------------------------------------------- model-free: letterbox arithmetic


def test_letterbox_factors_square_is_identity() -> None:
    scale, pad_x, pad_y = _letterbox_factors(1024, 1024, 1024)
    assert (scale, pad_x, pad_y) == (1.0, 0, 0)


def test_letterbox_factors_wide_pads_top_bottom() -> None:
    # 512 tall x 1024 wide -> scale 1.0 (width-bound), 256 px pad top & bottom.
    scale, pad_x, pad_y = _letterbox_factors(1024, 512, 1024)
    assert scale == pytest.approx(1.0)
    assert pad_x == 0
    assert pad_y == 256


# -------------------------------------------------------------------- model-free: the crop helper


def test_crop_masks_to_boxes_zeroes_outside() -> None:
    masks = np.ones((1, 10, 10), dtype=np.float32)
    boxes_grid = np.array([[2.0, 3.0, 6.0, 7.0]], dtype=np.float32)  # xyxy in grid coords
    out = _crop_masks_to_boxes(masks, boxes_grid)
    # Inside [3:7, 2:6] preserved, everything else exactly zero.
    assert np.all(out[0, 3:7, 2:6] == 1.0)
    zeroed = out[0].copy()
    zeroed[3:7, 2:6] = 0.0
    assert np.all(zeroed == 0.0)


# ------------------------------------------------------- model-free: the full decode on toy shapes


def test_decode_single_box_undoes_to_image_coords() -> None:
    out0 = _make_output0(8, [(100.0, 200.0, 40.0, 60.0, 0.9)])
    out1 = np.zeros((1, 32, 16, 16), dtype=np.float32)
    props = decode_fastsam(out0, out1, 1024, 1024, FastSAMConfig(), scale=1.0, pad_x=0, pad_y=0)
    assert len(props) == 1
    p = props[0]
    assert isinstance(p, Proposal)
    assert p.box == BBox(x=80, y=170, w=40, h=60)
    assert p.objectness == pytest.approx(0.9)
    assert p.mask is None  # return_masks defaults to False


def test_decode_confidence_filter_drops_low_conf() -> None:
    out0 = _make_output0(8, [(100.0, 100.0, 20.0, 20.0, 0.3)])  # below the 0.4 default
    out1 = np.zeros((1, 32, 16, 16), dtype=np.float32)
    props = decode_fastsam(out0, out1, 1024, 1024, FastSAMConfig(), scale=1.0, pad_x=0, pad_y=0)
    assert props == []


def test_decode_nms_suppresses_overlap_keeps_disjoint() -> None:
    # A and B overlap (IoU ~0.92 > 0.9) -> B (lower conf) suppressed; C is disjoint -> kept.
    out0 = _make_output0(
        8,
        [
            (100.0, 100.0, 50.0, 50.0, 0.9),  # A
            (101.0, 101.0, 50.0, 50.0, 0.8),  # B, near-duplicate of A
            (500.0, 500.0, 50.0, 50.0, 0.7),  # C, far away
        ],
    )
    out1 = np.zeros((1, 32, 16, 16), dtype=np.float32)
    props = decode_fastsam(out0, out1, 1024, 1024, FastSAMConfig(), scale=1.0, pad_x=0, pad_y=0)
    assert len(props) == 2
    # Ordered by descending objectness: A then C.
    assert [p.objectness for p in props] == pytest.approx([0.9, 0.7])


def test_decode_max_proposals_caps_by_objectness() -> None:
    out0 = _make_output0(
        8,
        [
            (100.0, 100.0, 20.0, 20.0, 0.6),
            (300.0, 300.0, 20.0, 20.0, 0.9),
            (500.0, 500.0, 20.0, 20.0, 0.7),
        ],
    )
    out1 = np.zeros((1, 32, 16, 16), dtype=np.float32)
    cfg = FastSAMConfig(max_proposals=2)
    props = decode_fastsam(out0, out1, 1024, 1024, cfg, scale=1.0, pad_x=0, pad_y=0)
    assert [p.objectness for p in props] == pytest.approx([0.9, 0.7])


def test_decode_undoes_letterbox_padding() -> None:
    # Wide letterbox: 1024x512 image, pad_y=256, scale=1.0. A box at input y must shift up by 256.
    scale, pad_x, pad_y = _letterbox_factors(1024, 512, 1024)
    out0 = _make_output0(8, [(200.0, 300.0, 40.0, 40.0, 0.9)])  # input coords (cx, cy)
    out1 = np.zeros((1, 32, 16, 16), dtype=np.float32)
    props = decode_fastsam(
        out0, out1, 1024, 512, FastSAMConfig(), scale=scale, pad_x=pad_x, pad_y=pad_y
    )
    assert len(props) == 1
    # cx=200 -> x1=180; cy=300 with pad_y=256 -> y1 = (280 - 256)/1 = 24.
    assert props[0].box == BBox(x=180, y=24, w=40, h=40)


def test_decode_masks_are_cropped_to_box_verified_shapes() -> None:
    """The acceptance test: synthetic [1,37,21504] / [1,32,256,256] tensors decode with masks.

    Uses the FULL verified 1024-operating-point shapes. A single prototype is set high everywhere
    so ``sigmoid(coeff @ protos) ~ 1`` across the whole grid (a mask that *bleeds*); the decode
    must crop it to the box, so mask pixels outside the box are exactly zero and inside are > 0.
    """
    anchors = 21504
    out0 = _make_output0(anchors, [(512.0, 512.0, 100.0, 80.0, 0.95)])
    # Drive mask coefficient 0 -> prototype 0, which is a constant +10 logit everywhere.
    out0[0, 5, 0] = 1.0
    out1 = np.zeros((1, 32, 256, 256), dtype=np.float32)
    out1[0, 0, :, :] = 10.0

    cfg = FastSAMConfig(return_masks=True)
    props = decode_fastsam(out0, out1, 1024, 1024, cfg, scale=1.0, pad_x=0, pad_y=0)
    assert len(props) == 1
    p = props[0]
    assert p.box == BBox(x=462, y=472, w=100, h=80)
    assert p.mask is not None
    assert p.mask.shape == (1024, 1024)  # original image resolution
    # Inside the box the mask is high; outside it is EXACTLY zero (crop-to-box, mandatory).
    assert float(p.mask[512, 512]) > 0.5
    outside = p.mask.copy()
    outside[p.box.y : p.box.y2, p.box.x : p.box.x2] = 0.0
    assert np.all(outside == 0.0)


# ------------------------------------------------------------------- real-model (skipped in CI)


@_needs_model
def test_real_model_predict_returns_proposals() -> None:
    inf = FastSAMInferencer(_MODEL_PATH, providers=_CPU)
    scene = np.full((480, 640, 3), 127, dtype=np.uint8)
    scene[100:200, 150:300] = (30, 200, 30)  # a distinct blob to propose on
    props = inf.predict(scene)
    assert isinstance(props, list)
    assert all(isinstance(p, Proposal) for p in props)
    for p in props:
        assert 0.0 <= p.objectness <= 1.0
        assert p.box.x2 <= 640 and p.box.y2 <= 480


@_needs_model
def test_real_model_propose_honours_config() -> None:
    inf = FastSAMInferencer(_MODEL_PATH, providers=_CPU)
    scene = np.full((512, 512, 3), 127, dtype=np.uint8)
    scene[50:150, 50:150] = (200, 30, 30)
    many = inf.propose(scene, FastSAMConfig(conf_thres=0.2, max_proposals=None))
    few = inf.propose(scene, FastSAMConfig(conf_thres=0.2, max_proposals=3))
    assert len(few) <= 3
    assert len(few) <= len(many)
