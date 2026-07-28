"""One ground-truth loader for every sidecar the harness reads (EVAL-02).

There is exactly **one** ``*.gt.json`` format in this repo, so there is exactly one loader.
Three kinds of sidecar share it:

* **Synthetic** (:mod:`object_search.synthetic.generator`) -- ``boxes``, a ``spec`` with the
  canvas ``width``/``height``, and an exact ``slice_metadata`` block. Ground truth is exact
  because the generator drew every instance.
* **Chipset** (EVAL-19, :mod:`object_search.synthetic.chipset`) -- ``boxes``, top-level
  ``width``/``height``, ``requested_n`` **and** ``achieved_n``, and an ``exemplar_index``.
* **Hand-labelled** demo/basketball frames, written by a human in the *same* format so the
  eval harness never needs a second reader.

Two honesty rules are load-bearing here:

1. **The achieved count is the number of boxes actually present, never the requested ``N``.**
   The chipset generator rejection-samples non-overlapping positions with an attempt cap, so
   it can place fewer than requested; the sidecar records ``achieved_n`` and *that* is the
   ground truth. This loader reads the boxes as truth and cross-checks ``achieved_n`` against
   ``len(boxes)``, raising on disagreement -- a sidecar that overstated its count would
   silently depress every method's recall (EVAL-19).
2. **No ground truth returns ``None``, not an empty list.** ``[]`` would mean "this image
   genuinely contains zero instances" (recall undefined), whereas ``None`` means "nobody has
   labelled this image" -- report coverage honestly rather than silently excluding it. This is
   the same not-assessed-versus-assessed-zero distinction the store enforces one layer up.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from object_search.provenance import repo_root
from object_search.schemas.geometry import BBox, ExemplarBox
from object_search.schemas.records import SliceMetadata

# The directories searched for ``<image_id>.gt.json``, in order. All three carry the identical
# sidecar format, which is the whole point of a single loader.
_GT_ROOTS: tuple[Path, ...] = (
    Path("assets") / "demo" / "chipset",
    Path("assets") / "demo" / "synthetic",
    Path("assets") / "demo" / "basketball",
    Path("assets") / "demo" / "textured",
)


class GroundTruth(BaseModel):
    """The exact instances in one image, plus what the benchmark needs to query it.

    A bare ``list[BBox]`` would drop two facts the harness cannot reconstruct: **which** box is
    the designated exemplar (so every method is queried with the *same* box and the comparison
    is not confounded, EVAL-19) and the **canvas size / slice metadata** the per-slice report
    breaks results down by (EVAL-10). Both live here so one loader serves both the metric layer
    and the benchmark.

    Attributes:
        image_id: The sidecar's stem, e.g. ``"chipset-01"``.
        boxes: Every ground-truth instance, in image pixels. **This is the truth**; the
            ``achieved_n`` field, when present, is only a cross-check.
        exemplar_index: Index into ``boxes`` of the designated query instance. ``0`` when the
            sidecar does not name one (deterministic given the generator's sort).
        exemplar_indices: Every designated exemplar index, in order. An **additive** field for
            the research datasets whose native annotations ship *several* exemplar boxes per image
            (FSCD-* provides three): the 3-exemplar run honours all of them and the 1-exemplar run
            takes the first. Empty by default; :attr:`effective_exemplar_indices` falls back to
            ``(exemplar_index,)`` so every existing single-exemplar sidecar is unchanged.
        width: Canvas width in pixels, or ``None`` if the sidecar did not record it.
        height: Canvas height in pixels, or ``None`` if the sidecar did not record it.
        slice_metadata: Per-slice descriptors (true instance count, scale range, ...). For the
            chipset the count is set from the achieved box count; for synthetic it is exact.
        source: Which root the sidecar came from -- ``"chipset"``, ``"synthetic"`` or ``"hand"``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    image_id: str = Field(min_length=1)
    boxes: tuple[BBox, ...] = Field(min_length=1)
    exemplar_index: int = Field(ge=0)
    exemplar_indices: tuple[int, ...] = ()
    width: int | None = None
    height: int | None = None
    slice_metadata: SliceMetadata = Field(default_factory=SliceMetadata)
    source: str

    @property
    def achieved_count(self) -> int:
        """The number of instances actually present -- always ``len(boxes)`` (EVAL-19)."""
        return len(self.boxes)

    @property
    def effective_exemplar_indices(self) -> tuple[int, ...]:
        """Designated exemplar indices, falling back to ``(exemplar_index,)`` when none listed.

        The single source every caller reads, so a single-exemplar sidecar (``exemplar_indices``
        empty) and a multi-exemplar research sidecar are queried through the same code path.
        """
        return self.exemplar_indices or (self.exemplar_index,)

    @property
    def exemplar(self) -> ExemplarBox:
        """The designated query box, so every method searches from the same exemplar."""
        return ExemplarBox(box=self.boxes[self.exemplar_index])

    def exemplar_at(self, count: int) -> ExemplarBox:
        """The exemplar for an ``count``-exemplar run: the first of the designated indices.

        The harness scores at 1 and 3 exemplars (a locked decision). Every method in this repo
        takes a single :class:`ExemplarBox`, so the ``count`` selects *which* designated box seeds
        the run -- ``1`` uses the first, ``3`` would use the third when three are listed -- rather
        than handing the method several boxes at once. ``count`` is clamped to the available
        indices so a 3-exemplar request on a 1-exemplar sidecar degrades to the first box.
        """
        indices = self.effective_exemplar_indices
        chosen = indices[min(count, len(indices)) - 1] if count >= 1 else indices[0]
        return ExemplarBox(box=self.boxes[chosen])


def _canvas_size(payload: dict[str, object]) -> tuple[int | None, int | None]:
    """Pull canvas ``(width, height)`` from a sidecar, from top level or a nested ``spec``."""
    for container in (payload, payload.get("spec")):
        if isinstance(container, dict):
            width, height = container.get("width"), container.get("height")
            if isinstance(width, int) and isinstance(height, int):
                return width, height
    return None, None


def _source_for(root_name: str) -> str:
    """Map a GT directory name to a short provenance tag."""
    if root_name == "chipset":
        return "chipset"
    if root_name == "synthetic":
        return "synthetic"
    if root_name == "textured":
        return "textured"
    return "hand"


def _parse_exemplar_indices(
    payload: dict[str, object], path: Path, n_boxes: int
) -> tuple[int, ...]:
    """Parse the optional additive ``exemplar_indices`` list, validating each is in range.

    Absent (the common single-exemplar case) yields ``()`` -- :attr:`GroundTruth.effective_
    exemplar_indices` then falls back to ``(exemplar_index,)``. Present, it must be a list of
    integers each inside ``[0, n_boxes)``, so a corrupt research sidecar fails loudly here rather
    than indexing out of bounds when the benchmark seeds an exemplar.
    """
    raw = payload.get("exemplar_indices")
    if raw is None:
        return ()
    if not isinstance(raw, list) or not all(isinstance(i, int) for i in raw):
        raise ValueError(f"{path}: exemplar_indices={raw!r} must be a list of integers")
    indices = tuple(int(i) for i in raw)
    if any(not (0 <= i < n_boxes) for i in indices):
        raise ValueError(f"{path}: exemplar_indices={indices} outside [0, {n_boxes})")
    return indices


def _parse_sidecar(path: Path, image_id: str, *, source_override: str | None = None) -> GroundTruth:
    """Parse one ``*.gt.json`` into a :class:`GroundTruth`, cross-checking the achieved count.

    ``source_override`` tags the provenance explicitly (research sidecars live under the
    ``datasets/`` tree, whose directory name would otherwise fall through to the ``"hand"``
    default and mislabel them in the report). Everything else -- the schema, the achieved-count
    cross-check, the exemplar-index validation -- is shared, so there is still exactly one parser
    (D-10): no research dataset gets a second GT reader.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    boxes = tuple(BBox.model_validate(box) for box in payload["boxes"])
    if not boxes:
        # An empty box list is not a valid label file: "no instances" is expressed by the
        # absence of a sidecar (loader returns None), never by an empty one.
        raise ValueError(f"{path} has an empty box list; a labelled image has >= 1 instance")

    # The achieved count is len(boxes). Where the sidecar also states it, disagreement means the
    # file is corrupt -- surface it rather than silently trusting one number over the other.
    stated = payload.get("achieved_n")
    if isinstance(stated, int) and stated != len(boxes):
        raise ValueError(
            f"{path}: achieved_n={stated} disagrees with {len(boxes)} boxes; corrupt sidecar"
        )

    exemplar_index = payload.get("exemplar_index", 0)
    if not isinstance(exemplar_index, int) or not (0 <= exemplar_index < len(boxes)):
        raise ValueError(f"{path}: exemplar_index={exemplar_index!r} is outside [0, {len(boxes)})")

    exemplar_indices = _parse_exemplar_indices(payload, path, len(boxes))

    width, height = _canvas_size(payload)

    slice_block = payload.get("slice_metadata")
    if isinstance(slice_block, dict):
        slice_metadata = SliceMetadata.model_validate(slice_block)
    else:
        # Chipset / hand sidecars carry no slice block; the exact count is still known.
        slice_metadata = SliceMetadata(true_instance_count=len(boxes))

    return GroundTruth(
        image_id=image_id,
        boxes=boxes,
        exemplar_index=exemplar_index,
        exemplar_indices=exemplar_indices,
        width=width,
        height=height,
        slice_metadata=slice_metadata,
        source=source_override or _source_for(path.parent.name),
    )


def chipset_image_ids() -> tuple[str, ...]:
    """Every chipset image id that has a committed sidecar, in canvas-size order.

    The benchmark's model-free CI subset (EVAL-19) sweeps this set; deriving it from the files
    on disk rather than hardcoding ``chipset-01..10`` means a regenerated set with a different
    length stays correct.
    """
    chipset_dir = repo_root() / _GT_ROOTS[0]
    if not chipset_dir.is_dir():
        return ()
    return tuple(sorted(path.name[: -len(".gt.json")] for path in chipset_dir.glob("*.gt.json")))


def textured_image_ids() -> tuple[str, ...]:
    """Every textured-regime image id with a committed sidecar (EVAL-20), sorted by id.

    Derived from the files on disk like :func:`chipset_image_ids`, so regenerating the set with a
    different size stays correct. The id prefix (``textured-plain-``, ``textured-varied-``,
    ``textured-cluttered-``) names the regime, which the report groups by.
    """
    textured_dir = repo_root() / "assets" / "demo" / "textured"
    if not textured_dir.is_dir():
        return ()
    return tuple(sorted(path.name[: -len(".gt.json")] for path in textured_dir.glob("*.gt.json")))


def scene_path(image_id: str) -> Path | None:
    """Absolute path to the scene image for ``image_id``, or ``None`` if not on disk.

    Searches the GT roots for ``<image_id>.png`` then ``<image_id>.jpg`` (basketball frames are
    JPEG). The benchmark loads the committed pixels rather than regenerating, so the number a
    method scores is the number a human would see in the UI.
    """
    for base in (repo_root() / r for r in _GT_ROOTS):
        for suffix in (".png", ".jpg"):
            candidate = base / f"{image_id}{suffix}"
            if candidate.is_file():
                return candidate
    return None


@cache
def load_ground_truth(image_id: str, root: Path | None = None) -> GroundTruth | None:
    """Load the ground truth for ``image_id``, or ``None`` if no sidecar exists.

    Searches the known GT roots (chipset, synthetic, basketball) for ``<image_id>.gt.json`` and
    returns the first match, parsed. Returning ``None`` for a missing label is deliberate: the
    benchmark reports coverage honestly instead of pretending an unlabelled image has no
    instances (EVAL-02).

    Args:
        image_id: The sidecar stem, e.g. ``"chipset-03"`` or ``"scatter-scaled"``.
        root: Optional single directory to search instead of the defaults -- for tests that
            write a sidecar into a ``tmp_path``.

    Returns:
        A frozen :class:`GroundTruth`, or ``None`` when no sidecar is found.

    Raises:
        ValueError: If a sidecar is found but is internally inconsistent (empty box list,
            ``achieved_n`` disagreeing with the boxes, or an out-of-range exemplar index).
    """
    roots = (root,) if root is not None else tuple(repo_root() / r for r in _GT_ROOTS)
    for base in roots:
        sidecar = base / f"{image_id}.gt.json"
        if sidecar.is_file():
            logger.debug("ground truth for {} loaded from {}", image_id, sidecar)
            return _parse_sidecar(sidecar, image_id)
    logger.debug("no ground truth sidecar for {} (searched {} root(s))", image_id, len(roots))
    return None


def load_research_ground_truth(sidecar_path: Path) -> GroundTruth | None:
    """Load a research-dataset ``*.gt.json`` sidecar by explicit path, tagged ``source="research"``.

    A thin wrapper over the single :func:`_parse_sidecar` (D-10): research datasets reach the
    benchmark by converting their native annotations into the **same** sidecar schema, so there is
    no second GT reader. The only difference from :func:`load_ground_truth` is that research
    sidecars are addressed by their absolute path under the gitignored ``datasets/`` tree (they are
    not in the committed ``_GT_ROOTS``) and are tagged ``"research"`` rather than falling through
    to the ``"hand"`` provenance default.

    Args:
        sidecar_path: Absolute path to a converted ``<image_id>.gt.json`` under ``datasets/``.

    Returns:
        A frozen :class:`GroundTruth` with ``source == "research"``, or ``None`` if the sidecar is
        not on disk (the raw research tree is gitignored and fetched, so absence is expected in a
        clean checkout).

    Raises:
        ValueError: If the sidecar is found but internally inconsistent (same guards as
            :func:`load_ground_truth`).
    """
    if not sidecar_path.is_file():
        logger.debug("no research ground truth sidecar at {}", sidecar_path)
        return None
    image_id = sidecar_path.name[: -len(".gt.json")]
    return _parse_sidecar(sidecar_path, image_id, source_override="research")
