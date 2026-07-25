"""``FastSAMInferencer`` -- class-agnostic region proposals from FastSAM-s (ONNX).

Method 5 (``propose-retrieve``, Phase 7) uses FastSAM in **"everything mode"** (single class
``{0: 'object'}``) as its proposal stage: one forward pass yields a few hundred class-agnostic
region boxes (and optionally their masks), which the retrieval stage then embeds and ranks. The
runtime package imports **nothing** from ``ultralytics``; it loads the exported ``.onnx`` under
ONNX Runtime. The AGPL-3.0 exporter lives only in the ``export`` pixi env (see
``docs/library-reviews/fastsam.md``).

The verified contract (exact numbers, per project constraint -- runtime-verified in
``.planning/research/MODELS.md`` and gated by the ``Trial`` verdict in
``docs/library-reviews/fastsam.md``)
------------------------------------------------------------------------------------------------
Input
  * name ``images``; dtype ``float32`` (``tensor(float)``); layout **NCHW**; shape
    ``[batch, 3, height, width]`` -- the channel dim is static 3, H/W are dynamic but the
    **operating point is 1024x1024**.
  * colour order **RGB** (scenes arrive BGR from OpenCV and are converted).
  * scale ``1/255``; **NO mean subtraction, NO std division** -- YOLO does no normalization, so
    ``mean`` is ``(0, 0, 0)`` and ``std`` is ``(1, 1, 1)`` (inert identities on the base path).
  * resize **letterbox** to 1024x1024: scale by ``min(1024/W, 1024/H)`` preserving aspect ratio,
    then pad to 1024x1024 with **fill 114** (the YOLO/Ultralytics grey), centred -- so the pad
    must be subtracted when mapping boxes back to image pixels.

Output (verified at 1024^2 / 640^2 / 512x768; the two channel dims are static, the rest dynamic)
  * ``output0`` f32 ``[batch, 37, anchors]`` -- per-anchor, **channels-first**. The 37 channels
    are ``[0:4]`` box ``(cx, cy, w, h)`` in *letterboxed input* pixels, ``[4]`` objectness /
    class-0 confidence, ``[5:37]`` 32 mask coefficients. At 1024^2, ``anchors == 21504``
    (``= 128^2 + 64^2 + 32^2``, strides 8/16/32).
  * ``output1`` f32 ``[batch, 32, mask_h, mask_w]`` -- 32 mask **prototypes** at stride 4. At
    1024^2, ``mask_h == mask_w == 256``.

The YOLOv8-seg output decoding, written out (this is the whole point of the docstring)
--------------------------------------------------------------------------------------
:func:`decode_fastsam` performs, in order:

1. **Transpose** ``output0[0]`` from ``[37, anchors]`` to ``[anchors, 37]``.
2. **Split** into ``boxes_xywh = [:, :4]``, ``conf = [:, 4]``, ``coeff = [:, 5:]`` (32 columns).
3. **Confidence filter**: keep anchors with ``conf > conf_thres`` (FastSAM default ``0.4``).
4. Convert the surviving boxes ``(cx, cy, w, h) -> (x1, y1, x2, y2)`` in letterboxed input pixels.
5. **NMS** on those boxes with ``conf`` at ``iou_thres`` (FastSAM default ``0.9`` -- deliberately
   loose: "everything mode" *wants* overlapping proposals; over-segmentation is handled by NMS
   *after* retrieval, not here). Tie-breaking is deterministic ``(-score, y1, x1)``.
6. **Masks (only when ``return_masks``)**: ``masks = sigmoid(coeff_survivors @
   protos.reshape(32, -1))`` reshaped to ``mask_h x mask_w``.
   Then **crop each mask to its own box** at proto-grid scale (:func:`_crop_masks_to_boxes`) --
   this is **mandatory**, because a raw prototype-combination mask bleeds well outside its
   detection. Each cropped mask is upsampled to
   image resolution, the letterbox padding is removed, and the final integer box is re-applied so
   pixels outside the box are **exactly** zero after interpolation.
7. **Undo the letterbox** on the boxes: subtract the pad, divide by the scale, clip to the
   original image, and emit each as a :class:`~object_search.schemas.BBox` (half-open, integer).

Only the **input** is validated at load (INFRA-09): ``anchors`` and the mask resolution are
data-/resolution-dependent, so no output-shape assertion is possible or wanted. The arithmetic in
steps 1-7 is pure and lives in module-level functions, so CI gates it with synthetic tensors of
the verified shapes -- no gitignored weight required.

Licence (carried from the library review): FastSAM is **AGPL-3.0** and the exported ``.onnx``
embeds that licence string. Private local use triggers nothing; publishing this repo or
network-exposing the API fires AGPL §13. The file is gitignored (INFRA-11) and arrives only via
``pixi run -e export export-fastsam``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field

from object_search.inference.onnx_inferencer import ONNXInferencer, ONNXInputSpec
from object_search.schemas import BBox

# The verified FastSAM operating point and channel layout (see the module docstring / MODELS.md).
FASTSAM_INPUT_SIZE: int = 1024  # letterbox target; H and W must be multiples of stride 32
FASTSAM_MASK_STRIDE: int = 4  # prototype grid is input_size / 4 (256 at 1024)
FASTSAM_N_COEFF: int = 32  # mask coefficients per anchor == prototype count
_BOX_COLS = slice(0, 4)  # output0 channels [0:4] -> cx, cy, w, h
_CONF_COL = 4  # output0 channel [4] -> objectness / class-0 confidence
_COEFF_COLS = slice(5, 37)  # output0 channels [5:37] -> 32 mask coefficients

# The full input contract as a typed object -- letterbox(1024, fill 114), RGB, /255, NO mean/std.
# The base ``preprocess`` executes this faithfully: its letterbox fill is 114 and its
# ``mean``/``std`` here are the identities, so the only normalization applied is ``* (1/255)``.
FASTSAM_INPUT_SPEC: ONNXInputSpec = ONNXInputSpec(
    input_name="images",
    dtype="tensor(float)",
    shape=(1, 3, FASTSAM_INPUT_SIZE, FASTSAM_INPUT_SIZE),
    layout="NCHW",
    color_order="RGB",
    scale=1.0 / 255.0,
    mean=(0.0, 0.0, 0.0),  # YOLO does no mean subtraction (inert identity)
    std=(1.0, 1.0, 1.0),  # YOLO does no std division (inert identity)
    resize="letterbox",
    interpolation="bilinear",
)


class FastSAMConfig(BaseModel):
    """Decoding parameters for the FastSAM proposal stage (frozen, one source of truth).

    Attributes:
        conf_thres: Keep anchors with objectness strictly greater than this. FastSAM's default is
            ``0.4``.
        iou_thres: NMS suppresses a later box whose IoU with a survivor exceeds this. FastSAM's
            default is ``0.9`` -- deliberately loose, because "everything mode" wants overlapping
            proposals and post-retrieval NMS handles duplicates.
        max_proposals: Optional cap on the number of proposals returned, keeping the highest
            objectness first. ``None`` returns all survivors.
        return_masks: When ``True``, decode ``output1`` into per-proposal masks (cropped to box);
            when ``False`` (the Milestone 1 default), the contract is boxes only and ``output1`` is
            skipped for speed. The robustness-backlog "background-masked embedding" flips this on.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    conf_thres: float = Field(default=0.4, ge=0.0, le=1.0)
    iou_thres: float = Field(default=0.9, ge=0.0, le=1.0)
    max_proposals: int | None = Field(default=None, ge=1)
    return_masks: bool = False


@dataclass(frozen=True, eq=False)
class Proposal:
    """One class-agnostic region proposal: a box, an optional mask, and an objectness score.

    ``eq=False`` because ``mask`` is a NumPy array (element-wise ``==`` has no single truth value);
    proposals are never compared for equality, they are ranked by ``objectness`` and deduplicated
    by box IoU.

    Attributes:
        box: The proposal region in original-image pixels (half-open, integer).
        mask: Optional binary/soft mask in original-image resolution, ``(H, W)`` float in
            ``[0, 1]``, **zero outside ``box``**. ``None`` when ``return_masks`` was off.
        objectness: FastSAM's class-0 confidence for this region, in ``[0, 1]``.
    """

    box: BBox
    mask: npt.NDArray[np.float32] | None
    objectness: float


# ------------------------------------------------------------------ pure decoding arithmetic


def _letterbox_factors(orig_w: int, orig_h: int, input_size: int) -> tuple[float, int, int]:
    """Return ``(scale, pad_x, pad_y)`` for a centred letterbox into ``input_size`` square.

    This mirrors :meth:`ONNXInferencer._resize_plan` for the ``letterbox`` policy exactly, so the
    factors used to *undo* the letterbox match the ones used to *apply* it. ``scale`` is the single
    (isotropic) resize factor; ``pad_x``/``pad_y`` are the left/top padding in input pixels.
    """
    scale = min(input_size / orig_w, input_size / orig_h)
    new_w = max(1, round(orig_w * scale))
    new_h = max(1, round(orig_h * scale))
    pad_x = (input_size - new_w) // 2
    pad_y = (input_size - new_h) // 2
    return scale, pad_x, pad_y


def _sigmoid(x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Numerically stable logistic sigmoid, returned as float32."""
    return (1.0 / (1.0 + np.exp(-x.astype(np.float32)))).astype(np.float32)


def _xywh_to_xyxy(boxes: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Convert ``(cx, cy, w, h)`` rows to ``(x1, y1, x2, y2)`` rows (all float, input coords)."""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    xyxy = np.empty_like(boxes)
    xyxy[:, 0] = cx - w / 2.0
    xyxy[:, 1] = cy - h / 2.0
    xyxy[:, 2] = cx + w / 2.0
    xyxy[:, 3] = cy + h / 2.0
    return xyxy


def _nms_xyxy(
    boxes_xyxy: npt.NDArray[np.float32],
    scores: npt.NDArray[np.float32],
    iou_thres: float,
) -> list[int]:
    """Deterministic greedy IoU NMS on float ``xyxy`` boxes; return kept row indices.

    Tie-breaking imposes a **total order** ``(-score, y1, x1)`` before the sweep, so equally-scoring
    boxes suppress in a caller-order-independent way -- the same reproducibility discipline as
    ``search/common/nms.py`` (PITFALLS.md 6.3). This is reimplemented on raw arrays rather than
    importing that helper because the inference layer must not depend on the search layer, and
    because NMS here runs on float input-space boxes before any :class:`BBox` is built.
    """
    n = boxes_xyxy.shape[0]
    if n == 0:
        return []
    areas = np.maximum(0.0, boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) * np.maximum(
        0.0, boxes_xyxy[:, 3] - boxes_xyxy[:, 1]
    )
    order = sorted(
        range(n),
        key=lambda i: (-float(scores[i]), float(boxes_xyxy[i, 1]), float(boxes_xyxy[i, 0])),
    )
    kept: list[int] = []
    suppressed: set[int] = set()
    for idx in order:
        if idx in suppressed:
            continue
        kept.append(idx)
        for other in order:
            if other in suppressed or other == idx:
                continue
            ix1 = max(boxes_xyxy[idx, 0], boxes_xyxy[other, 0])
            iy1 = max(boxes_xyxy[idx, 1], boxes_xyxy[other, 1])
            ix2 = min(boxes_xyxy[idx, 2], boxes_xyxy[other, 2])
            iy2 = min(boxes_xyxy[idx, 3], boxes_xyxy[other, 3])
            iw = max(0.0, float(ix2 - ix1))
            ih = max(0.0, float(iy2 - iy1))
            inter = iw * ih
            union = float(areas[idx] + areas[other] - inter)
            iou = inter / union if union > 0.0 else 0.0
            if iou > iou_thres:
                suppressed.add(other)
    return kept


def _crop_masks_to_boxes(
    masks: npt.NDArray[np.float32],
    boxes_grid_xyxy: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Zero every mask pixel outside its own box, at proto-grid scale. Mandatory, not optional.

    A prototype-combination mask ``sigmoid(coeff @ protos)`` is defined over the whole grid and
    bleeds outside the detection; cropping it to the box is what makes the mask actually correspond
    to *this* proposal. ``boxes_grid_xyxy`` are the survivor boxes expressed in proto-grid
    coordinates (input pixels divided by :data:`FASTSAM_MASK_STRIDE`).

    Args:
        masks: ``(K, mh, mw)`` soft masks in ``[0, 1]``.
        boxes_grid_xyxy: ``(K, 4)`` boxes in grid coordinates.

    Returns:
        A new ``(K, mh, mw)`` array with pixels outside each box set to ``0``.
    """
    k, mh, mw = masks.shape
    out = np.zeros_like(masks)
    for i in range(k):
        x1 = int(np.floor(boxes_grid_xyxy[i, 0]))
        y1 = int(np.floor(boxes_grid_xyxy[i, 1]))
        x2 = int(np.ceil(boxes_grid_xyxy[i, 2]))
        y2 = int(np.ceil(boxes_grid_xyxy[i, 3]))
        x1 = max(0, min(x1, mw))
        y1 = max(0, min(y1, mh))
        x2 = max(0, min(x2, mw))
        y2 = max(0, min(y2, mh))
        if x2 > x1 and y2 > y1:
            out[i, y1:y2, x1:x2] = masks[i, y1:y2, x1:x2]
    return out


def _mask_to_image(
    proto_mask: npt.NDArray[np.float32],
    box: BBox,
    orig_w: int,
    orig_h: int,
    scale: float,
    pad_x: int,
    pad_y: int,
    input_size: int,
) -> npt.NDArray[np.float32]:
    """Upsample one proto-scale mask to image resolution, remove the letterbox, re-crop to box.

    Upsamples the ``mh x mw`` proto mask to the ``input_size`` canvas, slices out the content
    region (the letterbox pad removed), resizes that to ``(orig_h, orig_w)``, then re-applies the
    integer ``box`` so pixels outside it are **exactly** zero even after bilinear interpolation.
    """
    upsampled = cv2.resize(proto_mask, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    new_w = max(1, round(orig_w * scale))
    new_h = max(1, round(orig_h * scale))
    content = upsampled[pad_y : pad_y + new_h, pad_x : pad_x + new_w]
    resized = cv2.resize(content, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR).astype(
        np.float32
    )
    cropped = np.zeros_like(resized)
    cropped[box.y : box.y2, box.x : box.x2] = resized[box.y : box.y2, box.x : box.x2]
    return cropped


def decode_fastsam(
    output0: npt.NDArray[np.float32],
    output1: npt.NDArray[np.float32],
    orig_w: int,
    orig_h: int,
    config: FastSAMConfig,
    *,
    scale: float,
    pad_x: int,
    pad_y: int,
    input_size: int = FASTSAM_INPUT_SIZE,
) -> list[Proposal]:
    """Decode raw FastSAM ONNX outputs into image-space :class:`Proposal` objects.

    Pure arithmetic (see the module docstring, steps 1-7): no ONNX session, no gitignored weight,
    so this is the CI-testable core. ``scale``/``pad_x``/``pad_y`` are the letterbox factors that
    map letterboxed-input pixels back to the original image.

    Args:
        output0: ``[1, 37, anchors]`` per-anchor predictions (channels-first).
        output1: ``[1, 32, mask_h, mask_w]`` mask prototypes. Consulted only when
            ``config.return_masks``.
        orig_w: Original image width in pixels.
        orig_h: Original image height in pixels.
        config: Decoding thresholds and the mask toggle.
        scale: The isotropic letterbox resize factor.
        pad_x: Left letterbox padding in input pixels.
        pad_y: Top letterbox padding in input pixels.
        input_size: The letterbox square size (1024 operating point).

    Returns:
        Proposals ordered by descending objectness, boxes clipped to the image, at most
        ``config.max_proposals`` of them.
    """
    # 1. Transpose [37, anchors] -> [anchors, 37].
    predictions = np.asarray(output0[0], dtype=np.float32).T  # (anchors, 37)

    # 2. Split into boxes / confidence / mask coefficients.
    boxes_xywh = predictions[:, _BOX_COLS]
    conf = predictions[:, _CONF_COL]
    coeff = predictions[:, _COEFF_COLS]

    # 3. Confidence filter.
    keep = conf > config.conf_thres
    if not bool(np.any(keep)):
        return []
    boxes_xywh = boxes_xywh[keep]
    conf = conf[keep]
    coeff = coeff[keep]

    # 4. (cx, cy, w, h) -> (x1, y1, x2, y2) in letterboxed input pixels.
    xyxy = _xywh_to_xyxy(boxes_xywh)

    # 5. NMS (deterministic tie-break), then optional top-K by objectness.
    survivors = _nms_xyxy(xyxy, conf, config.iou_thres)
    survivors.sort(key=lambda i: -float(conf[i]))
    if config.max_proposals is not None:
        survivors = survivors[: config.max_proposals]
    if not survivors:
        return []

    # 6. Masks (optional): sigmoid(coeff @ protos), reshape, crop-to-box at proto scale.
    masks_by_survivor: dict[int, npt.NDArray[np.float32]] | None = None
    if config.return_masks:
        _, n_proto, mask_h, mask_w = output1.shape
        protos = np.asarray(output1[0], dtype=np.float32).reshape(n_proto, -1)  # (32, mh*mw)
        surv_coeff = coeff[survivors]  # (K, 32)
        soft = _sigmoid(surv_coeff @ protos).reshape(len(survivors), mask_h, mask_w)
        boxes_grid = xyxy[survivors] / (input_size / mask_h)
        cropped = _crop_masks_to_boxes(soft, boxes_grid)
        masks_by_survivor = {s: cropped[j] for j, s in enumerate(survivors)}

    # 7. Undo the letterbox and emit image-space proposals.
    proposals: list[Proposal] = []
    for s in survivors:
        x1 = (float(xyxy[s, 0]) - pad_x) / scale
        y1 = (float(xyxy[s, 1]) - pad_y) / scale
        x2 = (float(xyxy[s, 2]) - pad_x) / scale
        y2 = (float(xyxy[s, 3]) - pad_y) / scale
        ix1 = max(0, min(round(x1), orig_w))
        iy1 = max(0, min(round(y1), orig_h))
        ix2 = max(0, min(round(x2), orig_w))
        iy2 = max(0, min(round(y2), orig_h))
        if ix2 - ix1 < 1 or iy2 - iy1 < 1:
            continue  # degenerate after clipping -- drop it rather than emit a 0-area box
        box = BBox(x=ix1, y=iy1, w=ix2 - ix1, h=iy2 - iy1)
        mask: npt.NDArray[np.float32] | None = None
        if masks_by_survivor is not None:
            mask = _mask_to_image(
                masks_by_survivor[s], box, orig_w, orig_h, scale, pad_x, pad_y, input_size
            )
        proposals.append(Proposal(box=box, mask=mask, objectness=float(conf[s])))
    return proposals


class FastSAMInferencer(ONNXInferencer[list[Proposal]]):
    """Load FastSAM-s (ONNX), validate its input contract at load, run it, decode proposals.

    The base :class:`ONNXInferencer` validates the input dtype/shape at construction (INFRA-09).
    This subclass supplies the FastSAM letterbox input spec and decodes the two raw outputs into
    a list of :class:`Proposal`. No output shape is asserted: the anchor count and mask resolution
    are resolution-dependent.

    Args:
        model_path: Path to ``fastsam_s.onnx`` (produced by ``pixi run -e export export-fastsam``;
            gitignored, AGPL-3.0). Missing -> ``FileNotFoundError``.
        config: Default decoding config used by :meth:`predict`. :meth:`propose` overrides it.
        providers: ONNX Runtime execution providers; ``None`` = the runtime default.
        intra_op_num_threads: Latency / CPU-contention control only (see base module docstring).
        inter_op_num_threads: Latency / CPU-contention control only.
    """

    def __init__(
        self,
        model_path: Path | str,
        config: FastSAMConfig | None = None,
        providers: list[str] | None = None,
        *,
        intra_op_num_threads: int | None = None,
        inter_op_num_threads: int | None = None,
    ) -> None:
        self.config = config or FastSAMConfig()
        super().__init__(
            model_path,
            self._decode_default,
            FASTSAM_INPUT_SPEC,
            providers=providers,
            intra_op_num_threads=intra_op_num_threads,
            inter_op_num_threads=inter_op_num_threads,
        )

    def _decode_default(
        self,
        outputs: list[npt.NDArray[np.generic]],
        orig_w: int,
        orig_h: int,
    ) -> list[Proposal]:
        """Post-processor for the base ``predict`` path: decode with :attr:`config`.

        Recomputes the letterbox factors from the original size (the letterbox is deterministic),
        so the base ``predict`` needs no access to the :class:`PreprocessInfo`.
        """
        out0, out1 = self._named_outputs(outputs)
        scale, pad_x, pad_y = _letterbox_factors(orig_w, orig_h, FASTSAM_INPUT_SIZE)
        return decode_fastsam(
            out0, out1, orig_w, orig_h, self.config, scale=scale, pad_x=pad_x, pad_y=pad_y
        )

    def _named_outputs(
        self, outputs: list[npt.NDArray[np.generic]]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Map ``outputs`` to ``(output0, output1)`` by name, falling back to positional order."""
        named = dict(zip(self.output_names, outputs, strict=True))
        try:
            out0 = named["output0"]
            out1 = named["output1"]
        except KeyError:  # a re-export renamed the outputs -- fall back to the documented order
            out0, out1 = outputs[0], outputs[1]
        return np.asarray(out0, dtype=np.float32), np.asarray(out1, dtype=np.float32)

    def propose(
        self,
        image: npt.NDArray[np.uint8],
        config: BaseModel,
    ) -> list[Proposal]:
        """Run FastSAM on one BGR scene and return proposals decoded with ``config``.

        This is the :class:`~object_search.search.proposals.ProposalBackend` entry point, so
        ``config`` is typed as :class:`~pydantic.BaseModel` (the backend-agnostic contract) and
        narrowed to :class:`FastSAMConfig` here -- the same idiom the registered ``search``
        functions use. It is the independently callable proposal unit: it knows nothing about
        exemplars or retrieval.

        Raises:
            TypeError: If ``config`` is not a :class:`FastSAMConfig`.
        """
        if not isinstance(config, FastSAMConfig):
            raise TypeError(
                f"FastSAMInferencer.propose expects a FastSAMConfig, got {type(config).__name__}"
            )
        orig_h, orig_w = int(image.shape[0]), int(image.shape[1])
        info = self.preprocess(image)
        outputs = self._session.run(None, {self._input_name: info.tensor})
        out0, out1 = self._named_outputs([np.asarray(o) for o in outputs])
        return decode_fastsam(
            out0,
            out1,
            orig_w,
            orig_h,
            config,
            scale=info.scale_x,
            pad_x=info.pad_x,
            pad_y=info.pad_y,
        )
