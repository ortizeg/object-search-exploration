"""The textured benchmark regimes (EVAL-20).

The chipset (EVAL-19) is deliberately NCC-favourable: identical, axis-aligned, fixed-scale,
low-texture chips. That is honest but incomplete -- `sparse-geo` abstains on it for want of
keypoints, and `dino-dense` / `propose-retrieve` never get appearance variation to work with. This
module adds the complementary regimes so the four-method comparison spans both worlds, with the
same exact-ground-truth-by-construction contract (no hand-labeling, no licensing).

Three regimes, each a stratum that isolates a strength:

* **plain** -- richly-textured emblem, fixed scale and rotation. Favours `sparse-geo`: the emblem
  carries many corners/blobs, so SIFT finds well over the 20-keypoint floor and the method engages
  instead of abstaining.
* **varied** -- the same emblem, but each instance is scaled (0.6-1.6x), rotated (+-35 deg), and
  brightness-jittered. Same object identity, varied appearance -- the regime that exercises the
  deep-feature methods.
* **cluttered** -- textured emblem with mild variation on a noisy gradient background, plus
  distractors (a *different* emblem, drawn but excluded from ground truth). A precision stress.

Two invariants are load-bearing and tested, exactly as for the chipset:

* **Strict non-overlap.** Every pair of ground-truth boxes has IoU 0 (rejection sampling with a
  hard attempt cap), so precision/recall are unambiguous.
* **The recorded count is the ACHIEVED count, never the requested N.** A sidecar that overstated
  the count would silently depress recall and make the benchmark lie.

The ground-truth box is the exact axis-aligned bounding box of the *transformed* emblem (from the
warped corners), so a rotated instance is boxed by its true extent, not its nominal size.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import numpy.typing as npt
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from object_search.schemas.geometry import BBox
from object_search.schemas.records import SliceMetadata
from object_search.synthetic.generator import SyntheticImage

# Master seed: every per-image, per-emblem, and per-instance draw derives from this so the whole
# set regenerates byte-identically.
_MASTER_SEED = 20260725
_PLACEMENT_MAX_ATTEMPTS = 3000
_INSET = 3
_PLACEMENT_GAP = 3
_KEYPOINT_FLOOR = 20  # a plain emblem must clear this or sparse-geo abstains and the set is moot.

Regime = Literal["plain", "varied", "cluttered"]

# Moderate canvas sizes: large enough for several instances, small enough that the learned methods
# (dino-dense especially) run quickly and the committed PNGs stay well under the 2 MB gate.
_REGIME_CANVASES: dict[Regime, tuple[tuple[int, int], ...]] = {
    "plain": ((640, 480), (800, 600), (1024, 768)),
    "varied": ((640, 480), (800, 600), (1024, 768)),
    "cluttered": ((640, 480), (800, 600), (1024, 768)),
}
_IMAGES_PER_REGIME = 16  # ~16 images x 3 regimes = 48; distinct seeds capture generation variance.
_EMBLEM_SIZE = 84


class EmblemSpec(BaseModel):
    """One richly-textured emblem -- the single "object class" for one image."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    seed: int
    size: int = Field(ge=16)


class TexturedImageSpec(BaseModel):
    """One textured benchmark image: canvas, emblem, count, and the regime's variation knobs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    image_id: str
    regime: Regime
    width: int = Field(ge=64)
    height: int = Field(ge=64)
    n_instances: int = Field(ge=1)
    emblem: EmblemSpec
    seed: int
    scale_min: float = Field(default=1.0, gt=0.0)
    scale_max: float = Field(default=1.0, gt=0.0)
    rotation_deg: float = Field(default=0.0, ge=0.0)
    brightness_jitter: float = Field(default=0.0, ge=0.0, lt=1.0)
    n_distractors: int = Field(default=0, ge=0)
    noisy_background: bool = False


def _rand_color(rng: np.random.Generator) -> tuple[int, int, int]:
    vals = rng.integers(0, 256, size=3)
    return (int(vals[0]), int(vals[1]), int(vals[2]))


def render_emblem(spec: EmblemSpec) -> npt.NDArray[np.uint8]:
    """Render a richly-textured emblem, deterministically, with enough corners/blobs for SIFT.

    Layered in a fixed RNG order (per EVAL-DESIGN): a 2-D colour gradient base, several overlapping
    filled primitives, high-contrast thin line/arc accents (what SIFT keys on), a fine speckle
    field, and an asymmetric corner mark so the emblem's orientation is unambiguous. Kept
    deterministic with ``cv2.LINE_8`` and a single seeded generator.
    """
    rng = np.random.default_rng(spec.seed)
    size = spec.size
    emblem = np.empty((size, size, 3), dtype=np.uint8)

    # 1. A 2-D colour gradient base (never flat -- gradients give the detector low-frequency
    #    structure and stop the guard from tripping).
    c00 = np.array(_rand_color(rng), dtype=np.float32)
    c10 = np.array(_rand_color(rng), dtype=np.float32)
    c01 = np.array(_rand_color(rng), dtype=np.float32)
    ys = np.linspace(0.0, 1.0, size, dtype=np.float32)[:, None]
    xs = np.linspace(0.0, 1.0, size, dtype=np.float32)[None, :]
    grad = (c00[None, None] * (1 - xs)[..., None] + c10[None, None] * xs[..., None]) * (1 - ys)[
        ..., None
    ] + c01[None, None] * ys[..., None]
    emblem[:, :] = np.clip(grad, 0, 255).astype(np.uint8)

    # 2. Several overlapping filled primitives.
    for _ in range(int(rng.integers(8, 13))):
        kind = int(rng.integers(0, 3))
        if kind == 0:
            x0, x1 = sorted(int(v) for v in rng.integers(0, size, size=2))
            y0, y1 = sorted(int(v) for v in rng.integers(0, size, size=2))
            cv2.rectangle(emblem, (x0, y0), (x1, y1), _rand_color(rng), -1, cv2.LINE_8)
        elif kind == 1:
            cx, cy = int(rng.integers(0, size)), int(rng.integers(0, size))
            radius = int(rng.integers(3, max(4, size // 3)))
            cv2.circle(emblem, (cx, cy), radius, _rand_color(rng), -1, cv2.LINE_8)
        else:
            pts = rng.integers(0, size, size=(3, 2)).astype(np.int32)
            cv2.fillPoly(emblem, [pts], _rand_color(rng), cv2.LINE_8)

    # 3. High-contrast thin line/arc accents -- the corners SIFT keys on.
    for _ in range(int(rng.integers(6, 10))):
        p0 = (int(rng.integers(0, size)), int(rng.integers(0, size)))
        p1 = (int(rng.integers(0, size)), int(rng.integers(0, size)))
        shade = 0 if rng.random() < 0.5 else 255
        cv2.line(emblem, p0, p1, (shade, shade, shade), int(rng.integers(1, 3)), cv2.LINE_8)

    # 4. A fine speckle field (blobs for the detector).
    speckle = rng.integers(-28, 29, size=(size, size, 3), dtype=np.int16)
    out: npt.NDArray[np.uint8] = np.clip(emblem.astype(np.int16) + speckle, 0, 255).astype(np.uint8)

    # 5. An asymmetric corner mark so orientation is unambiguous (aids rotation recovery, avoids a
    #    symmetric-degeneracy read).
    mark = max(6, size // 6)
    cv2.rectangle(out, (2, 2), (2 + mark, 2 + mark), (255, 255, 255), -1, cv2.LINE_8)
    cv2.rectangle(out, (2, 2), (2 + mark, 2 + mark), (0, 0, 0), 1, cv2.LINE_8)
    return out


def sift_keypoint_count(patch: npt.NDArray[np.uint8]) -> int:
    """Number of SIFT keypoints on a patch -- the gate that a textured emblem is findable."""
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    # cv2's type stubs omit the detector factory functions; sparse_geo.py ignores the same way.
    sift = cv2.SIFT_create()  # type: ignore[attr-defined]
    return len(sift.detect(gray, None))


def _transform_instance(
    emblem: npt.NDArray[np.uint8], scale: float, angle_deg: float, brightness: float
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    """Scale, brightness-adjust, and rotate the emblem; return ``(warped_bgr, mask)``.

    The mask marks the rotated-square footprint, so only emblem pixels are pasted and the corner
    triangles inside the axis-aligned bounding box keep the canvas background.
    """
    scaled = max(8, round(emblem.shape[0] * scale))
    resized = cv2.resize(emblem, (scaled, scaled), interpolation=cv2.INTER_AREA)
    bright = np.clip(resized.astype(np.float32) * brightness, 0, 255).astype(np.uint8)

    rot = cv2.getRotationMatrix2D((scaled / 2.0, scaled / 2.0), angle_deg, 1.0)
    cos, sin = abs(float(rot[0, 0])), abs(float(rot[0, 1]))
    out_w = int(scaled * cos + scaled * sin)
    out_h = int(scaled * sin + scaled * cos)
    rot[0, 2] += (out_w - scaled) / 2.0
    rot[1, 2] += (out_h - scaled) / 2.0
    warped = cv2.warpAffine(
        bright, rot, (out_w, out_h), flags=cv2.INTER_NEAREST, borderValue=(0, 0, 0)
    )
    solid = np.full((scaled, scaled), 255, dtype=np.uint8)
    mask = cv2.warpAffine(solid, rot, (out_w, out_h), flags=cv2.INTER_NEAREST, borderValue=0)
    return warped.astype(np.uint8), mask.astype(np.uint8)


def _overlaps(box: BBox, placed: list[BBox]) -> bool:
    gap = _PLACEMENT_GAP
    for other in placed:
        if not (
            box.x + box.w + gap <= other.x
            or other.x + other.w + gap <= box.x
            or box.y + box.h + gap <= other.y
            or other.y + other.h + gap <= box.y
        ):
            return True
    return False


def _fill_background(canvas: npt.NDArray[np.uint8], noisy: bool, rng: np.random.Generator) -> None:
    if not noisy:
        # A mild vertical gradient -- more realistic than flat white and a light stress on the
        # methods, without adding fake instances.
        top = np.array(_rand_color(rng), dtype=np.float32)
        bot = np.array(_rand_color(rng), dtype=np.float32)
        top = 0.5 * top + 0.5 * 255.0
        bot = 0.5 * bot + 0.5 * 255.0
        h = canvas.shape[0]
        ramp = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None, None]
        blended = top[None, None] * (1 - ramp) + bot[None, None] * ramp
        canvas[:, :] = np.clip(blended, 0, 255).astype(np.uint8)
        return
    base = np.array(_rand_color(rng), dtype=np.float32)
    base = 0.4 * base + 0.4 * 255.0
    h, w = canvas.shape[:2]
    # Coarse, low-frequency clutter: generate at ~1/8 resolution and upscale. This looks like real
    # background structure (mottling, soft blobs) rather than per-pixel static, and -- unlike static
    # -- it compresses to a small PNG so the committed asset stays under the large-file gate.
    small_h, small_w = max(8, h // 8), max(8, w // 8)
    coarse = rng.integers(-55, 56, size=(small_h, small_w, 3), dtype=np.int16).astype(np.float32)
    coarse_up = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_LINEAR)
    canvas[:, :] = np.clip(base[None, None] + coarse_up, 0, 255).astype(np.uint8)


def _paste(
    canvas: npt.NDArray[np.uint8],
    warped: npt.NDArray[np.uint8],
    mask: npt.NDArray[np.uint8],
    x: int,
    y: int,
) -> None:
    region = canvas[y : y + warped.shape[0], x : x + warped.shape[1]]
    sel = mask > 0
    region[sel] = warped[sel]


def generate_textured_image(spec: TexturedImageSpec) -> SyntheticImage:
    """Generate one textured image with exact, non-overlapping ground-truth boxes."""
    rng = np.random.default_rng(spec.seed)
    canvas = np.empty((spec.height, spec.width, 3), dtype=np.uint8)
    _fill_background(canvas, spec.noisy_background, rng)
    emblem = render_emblem(spec.emblem)

    placed: list[BBox] = []
    scales: list[float] = []
    rotations: list[float] = []
    attempts = 0
    while len(placed) < spec.n_instances and attempts < _PLACEMENT_MAX_ATTEMPTS:
        attempts += 1
        scale = float(rng.uniform(spec.scale_min, spec.scale_max))
        angle = float(rng.uniform(-spec.rotation_deg, spec.rotation_deg))
        brightness = float(rng.uniform(1.0 - spec.brightness_jitter, 1.0 + spec.brightness_jitter))
        warped, mask = _transform_instance(emblem, scale, angle, brightness)
        out_h, out_w = warped.shape[:2]
        if out_w + 2 * _INSET >= spec.width or out_h + 2 * _INSET >= spec.height:
            continue
        x = int(rng.integers(_INSET, spec.width - out_w - _INSET + 1))
        y = int(rng.integers(_INSET, spec.height - out_h - _INSET + 1))
        box = BBox(x=x, y=y, w=out_w, h=out_h)
        if _overlaps(box, placed):
            continue
        _paste(canvas, warped, mask, x, y)
        placed.append(box)
        scales.append(scale)
        rotations.append(angle)

    # Distractors: a different emblem, pasted but NOT recorded -- genuine false-positive bait. They
    # must also avoid the real instances so ground truth stays exact.
    if spec.n_distractors:
        distractor_spec = EmblemSpec(seed=spec.emblem.seed ^ 0x5F5F, size=spec.emblem.size)
        distractor = render_emblem(distractor_spec)
        d_placed = 0
        d_attempts = 0
        while d_placed < spec.n_distractors and d_attempts < _PLACEMENT_MAX_ATTEMPTS:
            d_attempts += 1
            scale = float(rng.uniform(spec.scale_min, spec.scale_max))
            angle = float(rng.uniform(-spec.rotation_deg, spec.rotation_deg))
            warped, mask = _transform_instance(distractor, scale, angle, 1.0)
            out_h, out_w = warped.shape[:2]
            if out_w + 2 * _INSET >= spec.width or out_h + 2 * _INSET >= spec.height:
                continue
            x = int(rng.integers(_INSET, spec.width - out_w - _INSET + 1))
            y = int(rng.integers(_INSET, spec.height - out_h - _INSET + 1))
            box = BBox(x=x, y=y, w=out_w, h=out_h)
            if _overlaps(box, placed):  # keep clear of real instances; distractors may touch
                continue
            _paste(canvas, warped, mask, x, y)
            d_placed += 1

    if len(placed) < spec.n_instances:
        logger.warning(
            f"{spec.image_id}: placed {len(placed)}/{spec.n_instances} instances after {attempts} "
            f"attempts; recording the achieved count as ground truth"
        )

    order = sorted(range(len(placed)), key=lambda i: (placed[i].y, placed[i].x))
    placed = [placed[i] for i in order]
    scales = [scales[i] for i in order]
    rotations = [rotations[i] for i in order]
    slice_metadata = SliceMetadata(
        true_instance_count=len(placed),
        instance_scale_min=min(scales) if scales else None,
        instance_scale_max=max(scales) if scales else None,
        rotation_min_deg=min(rotations) if rotations else None,
        rotation_max_deg=max(rotations) if rotations else None,
        clutter_level=1.0 if spec.noisy_background else 0.0,
    )
    return SyntheticImage(
        image=canvas, boxes=tuple(placed), spec=None, slice_metadata=slice_metadata
    )


def _regime_knobs(regime: Regime) -> dict[str, object]:
    if regime == "plain":
        return {
            "scale_min": 1.0,
            "scale_max": 1.0,
            "rotation_deg": 0.0,
            "brightness_jitter": 0.0,
            "n_distractors": 0,
            "noisy_background": False,
        }
    if regime == "varied":
        return {
            "scale_min": 0.6,
            "scale_max": 1.6,
            "rotation_deg": 35.0,
            "brightness_jitter": 0.25,
            "n_distractors": 0,
            "noisy_background": False,
        }
    return {
        "scale_min": 0.8,
        "scale_max": 1.3,
        "rotation_deg": 20.0,
        "brightness_jitter": 0.2,
        "n_distractors": 4,
        "noisy_background": True,
    }


def _build_specs() -> tuple[TexturedImageSpec, ...]:
    rng = np.random.default_rng(_MASTER_SEED)
    specs: list[TexturedImageSpec] = []
    for regime in ("plain", "varied", "cluttered"):
        canvases = _REGIME_CANVASES[regime]
        knobs = _regime_knobs(regime)
        for index in range(1, _IMAGES_PER_REGIME + 1):
            width, height = canvases[(index - 1) % len(canvases)]
            n_instances = int(rng.integers(8, 13))
            emblem_seed = int(rng.integers(0, 2**31))
            image_seed = int(rng.integers(0, 2**31))
            specs.append(
                TexturedImageSpec(
                    image_id=f"textured-{regime}-{index:02d}",
                    regime=regime,
                    width=width,
                    height=height,
                    n_instances=n_instances,
                    emblem=EmblemSpec(seed=emblem_seed, size=_EMBLEM_SIZE),
                    seed=image_seed,
                    **knobs,  # type: ignore[arg-type]
                )
            )
    return tuple(specs)


TEXTURED_SPECS: tuple[TexturedImageSpec, ...] = _build_specs()


def write_textured(out_dir: Path, *, force: bool = False, exemplar_index: int = 0) -> list[Path]:
    """Generate every textured image plus ``<image_id>.gt.json`` sidecars into ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for spec in TEXTURED_SPECS:
        image_path = out_dir / f"{spec.image_id}.png"
        if image_path.is_file() and not force:
            logger.info(f"{spec.image_id}: exists, skipping (use --force to overwrite)")
            written.append(image_path)
            continue
        result = generate_textured_image(spec)
        if not cv2.imwrite(str(image_path), result.image):
            raise OSError(f"failed to write textured image to {image_path}")
        sidecar = out_dir / f"{spec.image_id}.gt.json"
        payload = {
            "image": image_path.name,
            "width": spec.width,
            "height": spec.height,
            "regime": spec.regime,
            "seed": spec.seed,
            "emblem": spec.emblem.model_dump(mode="json"),
            "requested_n": spec.n_instances,
            "achieved_n": len(result.boxes),
            "exemplar_index": exemplar_index,
            "slice_metadata": result.slice_metadata.model_dump(mode="json"),
            "boxes": [box.model_dump(mode="json") for box in result.boxes],
        }
        sidecar.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        logger.info(
            f"{spec.image_id}: {spec.width}x{spec.height} [{spec.regime}], "
            f"{len(result.boxes)} instances (requested {spec.n_instances})"
        )
        written.append(image_path)
    return written
