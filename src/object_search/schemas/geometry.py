"""Geometry primitives shared by every layer of the system.

Box convention -- stated once, here, because every later module depends on it being
unambiguous. A :class:`BBox` stores ``(x, y, w, h)`` in **image pixels**, origin at the
**top-left** of the image, y growing downwards.

**(x, y) is the top-left corner and is inclusive; (x2, y2) is the bottom-right corner and is
EXCLUSIVE.** Therefore::

    x2 == x + w        y2 == y + h
    w  == x2 - x       h  == y2 - y
    area == w * h                      # no "+1" terms, anywhere

The interval is half-open, ``[x, x2) x [y, y2)`` -- exactly like a NumPy slice, so::

    crop = image[box.y : box.y2, box.x : box.x2]     # shape (box.h, box.w, ...)

is the crop with no off-by-one correction at any call site. A one-pixel box has
``w == h == 1``.

The half-open-versus-closed choice is the single most common source of off-by-one box bugs
in vision code, which is why this codebase has exactly one convention and it is the
NumPy-slice one. Anything that reads a box from an external source (a browser canvas, a
model output, a label file) converts *into* this convention at the boundary and never
carries a second convention inwards.

Every model here is frozen: a box that has been handed to a search method, persisted, or
rendered cannot be mutated behind the caller's back.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class BBox(BaseModel):
    """An axis-aligned bounding box in image pixels, half-open ``[x, x2) x [y, y2)``.

    This is *the* box type in the codebase. Detector output, ground truth, proposals,
    exemplars and rendered overlays all use it, so a box never needs converting between two
    internal representations.

    Attributes:
        x: Left edge, inclusive. ``>= 0``.
        y: Top edge, inclusive. ``>= 0``.
        w: Width in pixels. ``>= 1`` -- a zero-width box is not a box.
        h: Height in pixels. ``>= 1``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    x: int = Field(ge=0, description="Left edge in pixels, inclusive.")
    y: int = Field(ge=0, description="Top edge in pixels, inclusive.")
    w: int = Field(ge=1, description="Width in pixels.")
    h: int = Field(ge=1, description="Height in pixels.")

    # -- derived values are properties, never stored fields ---------------------------
    # A stored duplicate can disagree with its source after a partial edit; a property
    # cannot. The same rule governs LatencyBreakdown.total_ms and every Rating metric.

    @property
    def x2(self) -> int:
        """Right edge, **exclusive**: ``x + w``."""
        return self.x + self.w

    @property
    def y2(self) -> int:
        """Bottom edge, **exclusive**: ``y + h``."""
        return self.y + self.h

    @property
    def cx(self) -> float:
        """Centre x in continuous pixel coordinates: ``x + w / 2``."""
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        """Centre y in continuous pixel coordinates: ``y + h / 2``."""
        return self.y + self.h / 2.0

    @property
    def area(self) -> int:
        """Pixel count: ``w * h``. No ``+1`` term -- the box is half-open."""
        return self.w * self.h

    @property
    def xyxy(self) -> tuple[int, int, int, int]:
        """``(x, y, x2, y2)`` with ``x2``/``y2`` exclusive -- ready to slice with."""
        return (self.x, self.y, self.x2, self.y2)

    def iou(self, other: BBox) -> float:
        """Intersection-over-union with ``other``.

        Args:
            other: The box to compare against.

        Returns:
            IoU in ``[0.0, 1.0]``. ``0.0`` when the boxes do not overlap. Because the
            convention is half-open, two boxes that merely *touch* (``self.x2 == other.x``)
            do **not** overlap and score ``0.0``.
        """
        ix = max(self.x, other.x)
        iy = max(self.y, other.y)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        iw = ix2 - ix
        ih = iy2 - iy
        if iw <= 0 or ih <= 0:
            return 0.0
        intersection = iw * ih
        union = self.area + other.area - intersection
        return intersection / union

    def clipped_to(self, width: int, height: int) -> BBox:
        """Clip to an image of size ``width`` x ``height``.

        Args:
            width: Image width in pixels.
            height: Image height in pixels.

        Returns:
            A new box entirely inside ``[0, width) x [0, height)``.

        Raises:
            ValueError: If the box does not intersect the image at all. Raising is
                deliberate: silently returning a degenerate 1-pixel box would let a
                completely off-image detection survive as a plausible-looking match.
        """
        x = min(max(self.x, 0), width)
        y = min(max(self.y, 0), height)
        x2 = min(max(self.x2, 0), width)
        y2 = min(max(self.y2, 0), height)
        if x2 - x < 1 or y2 - y < 1:
            raise ValueError(
                f"BBox {self.xyxy} does not intersect an image of size {width}x{height}; "
                f"clipping would produce an empty box"
            )
        return BBox(x=x, y=y, w=x2 - x, h=y2 - y)


class ExemplarBox(BaseModel):
    """The box the user drew: the single positive example a search is seeded from.

    Deliberately a distinct type from :class:`BBox` rather than an alias. A function
    signature that says ``exemplar: ExemplarBox`` states *which* box it wants, so a scene
    box, a proposal and the query can never be transposed by accident. The
    ``SearchMethod`` protocol takes this type, not a bare box.

    Attributes:
        box: The drawn region, in scene-image pixels.
        label: Optional human label ("player", "bolt"). Carried for display and for the
            ground-truth workflow; no method is allowed to key behaviour off it, because
            the whole premise is exemplar-based search with no class vocabulary.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    box: BBox
    label: str | None = None


class Point(BaseModel):
    """A 2-D point in image pixels, ``float`` because keypoints are sub-pixel.

    Keypoint detectors refine to sub-pixel precision (SIFT) or return integers
    (SuperPoint returns ``int64``). Storing floats accommodates both without a second
    type; the inferencer docstring records which precision its backend actually delivers.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    x: float
    y: float
